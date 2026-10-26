Reading additional input from stdin...
OpenAI Codex v0.155.1
--------
workdir: /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
model: gpt-6-astra
provider: openai
approval: never
sandbox: read-only
reasoning effort: medium
reasoning summaries: none
session id: 01a0d4f0-11b1-7c31-a49e-c5e90cef0c7b
--------
user
Read /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md and review that CLAIM adversarially, verifying every file:line against this worktree. Verdict SHIP/HOLD first, then numbered findings.
codex
I’ll read the claim and repository rules, then verify each cited location and check whether the evidence supports the claim.
exec
/usr/bin/bash -lc 'pwd; git status --short; cat CLAUDE.md; cat docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md' in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 671ms:
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 M config/wb/campaign/spectral_t63_cam6.yaml
?? .venv
?? docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md
?? docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md
?? docs/wb/reviews/2026-09-24_clubb_liquid_spectral_glm.md
?? main
# legoESM Claude Memory

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
- **MPI AD**: `global_sum_mpi` (allreduce SUM) full VJP. MPI halo: `_sendrecv_vjp` custom_vjp. `fix_mass`/`zero_mean_tendency` flow grads via global reductions. `global_max_mpi`/`global_min_mpi` NOT diff — keep out of losses.
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

MECHANICAL CHECK, end of every task: print list of choices made, each one line,
marked ASKED or UNASKED. Empty UNASKED list = target. Anything on UNASKED list
offered for revert in same message. Worked example, five from 2026-08-21 that
prompted this rule: routing each case's droplet number into single-column arm;
allocating zero graupel field in every column running microphysics; which
campaign cases got registered and using ARM cumulus case as stand-in for another
campaign's regime; pinning forcing downloads to fixed versions and checksums;
turning missing variable name into hard error. Every one defensible. None asked.
That the error.

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
  water source; they live in `_run_per_step` while century run `_run_mpas`,
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
  three in one day): quoted `_run_mpas` "turbulent surface fluxes are
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
- **MPI halo AD**: all `sendrecv` via `_sendrecv_vjp` (`@jax.custom_vjp` in `halo_exchange.py`). Only `allreduce(SUM)` AD-safe; `MAX`/`MIN`/`allgather`/`bcast` = diagnostics only.
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
- **No runtime outputs in git.** `diagnostics/`, `logs/`, `**/logs/`, `results/`, `checkpoints/`, `output/`, `*.zarr`/`*.nc`, root `*.png`/`*.pdf`/`*.svg` gitignored. Visual-regression baselines stay in LOCAL working copies, regenerated on demand — not tracked. Never `git add -f` a runtime artifact.
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
7. Stop when clean or after 30 iter# CLAIM REVIEW (before code): CLUBB's liquid partition on the spectral training lane

Worktree: this directory (branch wb/cam6-baseline, just merged origin/main). No code written yet.
Be adversarial: attack the premise, the mechanism, the conservation argument, the scope.

## Situation
Owner directive 2026-09-24: the WeatherBench spectral training deck
(config/wb/campaign/spectral_t63_cam6.yaml) must be CONSISTENT WITH THE MPAS PRODUCTION DECK
(config/amip/amip_production.yaml) on three settings, which production now carries:
  1. clubb_liquid_partition: true   (CLUBB's diagnosed liquid handed to the host cloud water)
  2. hard_saturation_adjustment: false  (no condensation step after the physics; CAM6 clubb_do_liqsupersat=.false.)
  3. hard_sat_ice_curve: false
(2) and (3) are deck rows and are already flipped on the WB deck (3 was never set; the spectral
lane also has no post-step ice curve). (1) is BLOCKED BY CODE on this lane, twice:
 - turbulence/integration.py:~545 raises NotImplementedError for any non-MPAS model_type:
   "liquid_partition returns a cloud-liquid tendency (TurbulenceOutput.dq_c_dt) that the
   'spectral_pe' turbulence lane does not route to a condensate tracer, so the liquid the closure
   removed from vapour would be destroyed."
 - combined.py:251 refuses cld_macmic_num_steps>1 outside ("hydrostatic","mpas"); combined.py:281
   requires cld_macmic_num_steps>=2 whenever the partition is on, because the closure REPLACES the
   host cloud water and the microphysics must run on the replaced value; the spectral combined
   path (_make_spectral_pe_combined, ~L1385) evaluates modules in PARALLEL on one state and sums
   tendencies, while the macmic sub-cycle lives only in _make_hydrostatic_combined (~L606).

## THE CLAIM
C1. Piece A (small): the spectral turbulence bridge routes TurbulenceOutput.dq_c_dt into the q_c
    tracer tendency exactly as the MPAS bridge does (turbulence/integration.py ~L847/~L1022),
    refusing a state with no q_c tracer. Vapour tendency already has that liquid removed, so water
    is conserved; no new sign, no clip.
C2. Piece B (larger): the ordering requirement is REAL (autoconversion ~ q_c^2.5 on pre-exchange
    liquid), so the partition cannot ship on the spectral lane by merely lifting the gate. Either
    the macmic sequential sub-cycle is ported to the spectral combined path (apply each module's
    increment to the spectral state between modules -- extra spectral<->grid transforms per
    sub-step, and the WB gradient's memory arena is ALREADY 643 GiB on one GPU before this), or
    the partition stays refused on this lane and the WB deck documents it as an open departure.
C3. Scope: nothing about the MPAS/hydrostatic lanes changes; the gate at combined.py:281 keeps its
    meaning; (2) and (3) are already consistent.

## Questions
1. Is C2's ordering requirement really binding on the spectral lane, or is there a cheaper
   faithful ordering (e.g. evaluating turbulence first and feeding its updated q_c only to the
   microphysics within the existing parallel accumulate)?
2. What is the memory and compile cost of a sequential sub-cycle inside a rollout step that is
   already remat'd with nothing_saveable, at T63/L32?
3. Does piece A alone (with the partition still refused) have any value, or is it dead code?
4. What must be MEASURED before any of this lands?
Verdict SHIP/HOLD first, then numbered findings with file:line.

exec
/usr/bin/bash -lc "sed -n '300,700p' CLAUDE.md" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 400ms:
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
  water source; they live in `_run_per_step` while century run `_run_mpas`,
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
  three in one day): quoted `_run_mpas` "turbulent surface fluxes are
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

exec
/usr/bin/bash -lc "rg --files -g AGENTS.md -g '*combined.py' -g '*integration.py' | head -70; git worktree list; nl -ba docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md; rg -n 'liquid_partition|cld_macmic_num_steps|hard_sat' config/wb/campaign/spectral_t63_cam6.yaml config/amip/amip_production.yaml src/legoesm/atmosphere" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 exited 2 in 861ms:
AGENTS.md
packages/ocean/legoesm/ocean/physics/lateral_mixing/integration.py
packages/ocean/legoesm/ocean/physics/combined.py
packages/ocean/legoesm/ocean/physics/convection/integration.py
tests/unit/test_physics_combined.py
tests/unit/test_ryf_pipeline_integration.py
tests/unit/test_land_ml_checkpoint_integration.py
packages/ocean/legoesm/ocean/physics/surface_forcing/integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/fesom_integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/mpas_integration.py
packages/ocean/legoesm/ocean/physics/vertical_mixing/integration.py
tests/ocean/unit/test_vmix_mpas_integration.py
tests/ocean/unit/test_eke_3d_integration.py
tests/ocean/unit/test_tracer_time_integration.py
tests/ocean/unit/test_catke_integration.py
tests/ocean/unit/test_tke_integration.py
tests/ocean/fidelity/test_mitgcm_integration.py
tests/land/integration/test_canopy_sif_integration.py
tests/run/test_offline_amip_forcing_integration.py
tests/run/test_correction_e2e_integration.py
tests/run/test_cmip_compare_integration.py
packages/core/legoesm/timestepping/integration.py
tests/atmosphere/hydrostatic/unit/test_aerosol_ccn_combined.py
packages/atmosphere/legoesm/atmosphere/physics/radiation/integration.py
packages/atmosphere/legoesm/atmosphere/physics/convection/integration.py
packages/atmosphere/legoesm/atmosphere/physics/combined.py
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py
packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/integration.py
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM                                                                                    13dfa47c9 [fix/wb-stability-restore]
/tmp/claude-200166/-work-bd1083-b309178-diffESM-legoesm-pg-legoESM/6139b175-a345-4fb0-abf8-a003edfc5b04/scratchpad/wt_base_cldfrc  f7c676cdb (detached HEAD) prunable
/tmp/claude-200166/-work-bd1083-b309178-diffESM-legoesm-pg-legoESM/6139b175-a345-4fb0-abf8-a003edfc5b04/scratchpad/wt_hp_base      270eec6ff (detached HEAD) prunable
/tmp/claude-200166/-work-bd1083-b309178-diffESM-legoesm-pg-legoESM/6139b175-a345-4fb0-abf8-a003edfc5b04/scratchpad/wt_preconsol    6a1c40f65 (detached HEAD) prunable
/work/bd1083/b309178/diffESM/_lever1_wt                                                                                            fc12673b8 [scaling/mpas-wide-halo-scan]
/work/bd1083/b309178/diffESM/legoesm_main_wt                                                                                       34f91d182 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_nature_base                                                                                   ee7a95871 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_nature_wt                                                                                     f278bf889 [scaling/nature-figure-2026-09]
/work/bd1083/b309178/diffESM/legoesm_pg/_prfix/i1464                                                                               6fcca5b94 [fix/1464-wire-neural-surface-drag]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/agent-a1d51538e3e836840                                          a3658ede6 [feat/mpas-ocean-spmd-lane]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/agent-ac78430029eb34e21                                          b8f2a51ff [feat/tripole-orca12-spmd-seaice]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/agent-af831759c838077a7                                          762b72e30 [worktree-agent-af831759c838077a7]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/agent-afd64072205022acb                                          50dc74a1f [worktree-agent-afd64072205022acb]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/bci-1028                                                         6158b06b5 [fix/1028-bci-growth]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/levante-ocean                                                    c9e90486d [worktree-levante-ocean]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/pd-1028                                                          6c7325e62 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/.claude/worktrees/scaling-campaign                                                 e299f69c4 [main]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/wt-loopclose                                                                       9be3aef70 [ocean/benchmark-figures]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM/wt_wbmem                                                                           51ae7b92a (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_ab_oldbake                                                                         cff706771 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_amip3                                                                              0af2456a0 [amip-climateeval-obs-refs]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_bomex                                                                              bfaee4557 [bomex-clubb-tune]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_campaign                                                                           d290146b4 [surface-stability-flag]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_clubbarm                                                                           cafae4430 [clubb-tuned-arm]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_clubbparams                                                                        513d9e1fd [clubb-params]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_cmorvars                                                                           179fb0c62 [cmor-cloud-vars]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_floorarm                                                                           e5cdb189a [floor-bound]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_ice                                                                                b292fc271 [clearsky-morrison-ice]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_lockex_pr                                                                          9566c1dad [lockex-rpe-dycore-fixes]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_mergemain                                                                          0cceb9c40 [sfs-mainmerge]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_mpicmor                                                                            1111a506e [mpi-cmor-feed]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_prfold                                                                             f09371053 [pr/microphysics-nr-ceiling]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_prfold2                                                                            d810ffc27 [pr/config-knobs]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_prfold3                                                                            6363d26bc [pr/land-interface-flux]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_utls                                                                               b2993dc91 [utls-cmor-clamp]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_vadvarm                                                                            97e70d3b2 [vadv-floor]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_vertadv                                                                            dbbd529d6 [utls-vert-adv]
/work/bd1083/b309178/diffESM/legoesm_pg/legoESM_vertadv_base                                                                       73159c7d0 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/legoesm_pe_slice                                                                           57b0c7629 [feat/spectral-pe-complex64]
/work/bd1083/b309178/diffESM/legoesm_pg/legoesm_pr_channels                                                                        c0118e095 [perf/nccl-channels-16]
/work/bd1083/b309178/diffESM/legoesm_pg/legoesm_pr_precision                                                                       41206b6ca [feat/precision-consistency]
/work/bd1083/b309178/diffESM/legoesm_pg/legoesm_pr_spectral                                                                        f4a14c551 [feat/spectral-sw-complex64]
/work/bd1083/b309178/diffESM/legoesm_pg/wt-amipwb                                                                                  c380ec1eb [feat/wb-amip-start]
/work/bd1083/b309178/diffESM/legoesm_pg/wt-amipwb-head                                                                             22e523d3e (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt-amipwb-opfix                                                                            22e523d3e (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt-nanprobe                                                                                c380ec1eb (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_arms                                                                                    93e595ffe (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_arms172                                                                                 172cf600b (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6                                                                                    6323e04ef [feat/cam6-suite]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6_cadence                                                                            3aed6fddd [feat/cam6-cadence]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6_cldfrc                                                                             7f5feb1c0 [feat/cam6-cldfrc]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6_default                                                                            5b395f486 [feat/cam6-production-default]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6_gwdflux                                                                            08a7fc418 [feat/cam6-gwd-flux]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6_merge                                                                              fe9adcc7f [merge/cam6-main]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6_repro                                                                              ebd7c08ef (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cam6_zm                                                                                 b731b24ee [feat/cam6-zm]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_cloudreview                                                                             0f8996933 [fix/cloud-cover-audit-2026-09]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_clubbliq                                                                                25c41e36d [feat/clubb-liquid-consistency]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_deficit                                                                                 3e42d8c77 [diag/lwp-timeseries]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_desert                                                                                  8f96cbe0b [fix/land-forcing-height]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_desert_ctl                                                                              30eed2c41 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_era5flux                                                                                9ca174461 [feat/wb-era5-surface-fluxes-spatial-embedding]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_frazil                                                                                  19f13007d [feat/ocean-frazil-closure]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_hc_pin                                                                                  0a23a8260 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_hc_pin2                                                                                 1a7685a4f (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_hires_compare                                                                           ef56edc02 [feat/hires-lane-compare]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_hires_ocean                                                                             ea436e28b [feat/hires-ocean-fesom-forca20]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_ic                                                                                      1d637fd31 [feat/hires-ic-from-fesom]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_land_accel                                                                              97dba6b35 [feat/land-packed-cadence]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_main_gate                                                                               6dded580e (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_mpas_run                                                                                a3658ede6 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_mpiparity                                                                               07b37abec [fix/mpas-multirank-momentum]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_parity                                                                                  83f3cb398 [feat/hires-lane-parity]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_parity2                                                                                 ddbd87538 [feat/hires-lane-parity-2]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_pr_dt112                                                                                9bc1fb611 [feat/amip-dt112]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_pr_res6                                                                                 643454dd4 [feat/amip-res6-default]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_pr_rhebc                                                                                566554cf3 [feat/amip-land-subcloud-rhebc]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_pr_soften                                                                               ac1af2c0a [fix/rhebc-soften-065]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_prefix_vac                                                                              07b37abec (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_probe_norad                                                                             61d078207 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_run2                                                                                    61d078207 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_run3                                                                                    f099f6e58 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_utls                                                                                    b3223aea1 [fix/utls-warm-bias-2026-09]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_utls_ctl                                                                                8065cb0e9 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_utls_prod                                                                               8aa15507e [feat/raised-lid-l36]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_val_gate                                                                                a30437103 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_watervapor                                                                              8d0f59350 [fix/tropical-moisture-2026-09]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6                                                                                  afda2fb8e [wb/cam6-baseline]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6_mem                                                                              a0038f279 (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbfix                                                                                   c29ed559a [fix/wb-checkpoint-scan]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbflux                                                                                  f09ca2403 [chore/wb-train-link-launcher]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbland                                                                                  5df9d37f6 [wb/classical-land-frac]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wblr                                                                                    864d6b840 [chore/wb-link-lr-override]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbmem                                                                                   39969532c [probe/wb-memory-ablation]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbtrain                                                                                 594cabf54 [wb/train-three-2026-09]
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wv_base                                                                                 b636d812a (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wv_ledger                                                                               196af6b6d (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pg/wt_wv_main                                                                                 93e595ffe (detached HEAD)
/work/bd1083/b309178/diffESM/legoesm_pr_edgeorder                                                                                  080860639 [perf/mpas-edge-hilbert-relabel]
/work/bd1083/b309178/diffESM/legoesm_scaling_wt                                                                                    4590d4d3b [scaling/precision-scaling-laws]
/work/bd1083/b309178/diffESM/legoesm_wt_overlap                                                                                    29ffbcfa0 [scaling/halo-compute-overlap]
/work/bd1083/b309178/diffESM/legoesm_wt_prdoc                                                                                      e200ab579 [scaling/assessment-doc-2026-08-16]
/work/bd1083/b309178/diffESM/legoesm_wt_prepack                                                                                    a96a62591 (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/1444                                                                                          c8d9214ae [feat/omip-core2-restart]
/work/bd1083/b309178/diffESM/prmerge/1603                                                                                          a98019fb6 (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/1605                                                                                          23d35bb90 (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/1610                                                                                          c07201bba [merge1638]
/work/bd1083/b309178/diffESM/prmerge/1622                                                                                          9efecd3cf (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/1623                                                                                          b996fe281 (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/1624                                                                                          1d5b19a1d (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/1625                                                                                          787967f66 [merge1642]
/work/bd1083/b309178/diffESM/prmerge/1630                                                                                          ceb91adbe [merge1630]
/work/bd1083/b309178/diffESM/prmerge/1640                                                                                          b7d277a50 [merge1640]
/work/bd1083/b309178/diffESM/prmerge/1644                                                                                          ad270a03d [merge1644]
/work/bd1083/b309178/diffESM/prmerge/1645                                                                                          fe32080d6 [merge1645]
/work/bd1083/b309178/diffESM/prmerge/1646                                                                                          eaaee975c [merge1646]
/work/bd1083/b309178/diffESM/prmerge/1648                                                                                          6010fc82e [merge1648]
/work/bd1083/b309178/diffESM/prmerge/fix                                                                                           7e07e1f67 [work/issue-sweep]
/work/bd1083/b309178/diffESM/prmerge/fix1666                                                                                       1b66fb743 [fix/1666-spmd-profile-slice]
/work/bd1083/b309178/diffESM/prmerge/fix1667                                                                                       675188a7f (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/i1028                                                                                         fc9ba3bae [feat/1028-cube-scale-selective-diffusion]
/work/bd1083/b309178/diffESM/prmerge/i1663                                                                                         b1a7b4fdf (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/i1665fix                                                                                      927da0e5e [fix/1665-precision-mixed-gate]
/work/bd1083/b309178/diffESM/prmerge/i1670                                                                                         d4142311d [feat/scm-les-tuned-turbulence-defaults]
/work/bd1083/b309178/diffESM/prmerge/i1675lift                                                                                     cd3fe374e [feat/1675-mixed-lift]
/work/bd1083/b309178/diffESM/prmerge/i1675s2                                                                                       f852af39f [feat/1675-site2-lmip-storage-dtype]
/work/bd1083/b309178/diffESM/prmerge/i1675s3                                                                                       8f86b2ef4 [feat/1675-site3-sota-mass]
/work/bd1083/b309178/diffESM/prmerge/i1677                                                                                         9e0795ac8 [pr1677]
/work/bd1083/b309178/diffESM/prmerge/main                                                                                          5863ee73c (detached HEAD)
/work/bd1083/b309178/diffESM/prmerge/scmrce                                                                                        2c5a707b3 [feat/scm-rce-campaign-f0]
/work/bd1083/b309178/diffESM/wt-baro-pr                                                                                            2189ae7dc [ocean/barotropic-centred-window]
/work/bd1083/b309178/diffESM/wt-clubb-1508                                                                                         5383b4ec7 [fix/clubb-sfc-varnce-1508]
/work/bd1083/b309178/diffESM/wt-ocean-baro                                                                                         d47964782 (detached HEAD)
/work/bd1083/b309178/diffESM/wt-ocean-base                                                                                         722725e0b (detached HEAD)
/work/bd1083/b309178/diffESM/wt-ocean-grid                                                                                         ee0a0dbc0 [ocean/grid-intercomparison-pr]
/work/bd1083/b309178/diffESM/wt-prefix-mem                                                                                         aec37d923 (detached HEAD)
/work/bd1083/b309178/diffESM/wt-wbdiag                                                                                             c9f74f931 (detached HEAD)
     1	# CLAIM REVIEW (before code): CLUBB's liquid partition on the spectral training lane
     2	
     3	Worktree: this directory (branch wb/cam6-baseline, just merged origin/main). No code written yet.
     4	Be adversarial: attack the premise, the mechanism, the conservation argument, the scope.
     5	
     6	## Situation
     7	Owner directive 2026-09-24: the WeatherBench spectral training deck
     8	(config/wb/campaign/spectral_t63_cam6.yaml) must be CONSISTENT WITH THE MPAS PRODUCTION DECK
     9	(config/amip/amip_production.yaml) on three settings, which production now carries:
    10	  1. clubb_liquid_partition: true   (CLUBB's diagnosed liquid handed to the host cloud water)
    11	  2. hard_saturation_adjustment: false  (no condensation step after the physics; CAM6 clubb_do_liqsupersat=.false.)
    12	  3. hard_sat_ice_curve: false
    13	(2) and (3) are deck rows and are already flipped on the WB deck (3 was never set; the spectral
    14	lane also has no post-step ice curve). (1) is BLOCKED BY CODE on this lane, twice:
    15	 - turbulence/integration.py:~545 raises NotImplementedError for any non-MPAS model_type:
    16	   "liquid_partition returns a cloud-liquid tendency (TurbulenceOutput.dq_c_dt) that the
    17	   'spectral_pe' turbulence lane does not route to a condensate tracer, so the liquid the closure
    18	   removed from vapour would be destroyed."
    19	 - combined.py:251 refuses cld_macmic_num_steps>1 outside ("hydrostatic","mpas"); combined.py:281
    20	   requires cld_macmic_num_steps>=2 whenever the partition is on, because the closure REPLACES the
    21	   host cloud water and the microphysics must run on the replaced value; the spectral combined
    22	   path (_make_spectral_pe_combined, ~L1385) evaluates modules in PARALLEL on one state and sums
    23	   tendencies, while the macmic sub-cycle lives only in _make_hydrostatic_combined (~L606).
    24	
    25	## THE CLAIM
    26	C1. Piece A (small): the spectral turbulence bridge routes TurbulenceOutput.dq_c_dt into the q_c
    27	    tracer tendency exactly as the MPAS bridge does (turbulence/integration.py ~L847/~L1022),
    28	    refusing a state with no q_c tracer. Vapour tendency already has that liquid removed, so water
    29	    is conserved; no new sign, no clip.
    30	C2. Piece B (larger): the ordering requirement is REAL (autoconversion ~ q_c^2.5 on pre-exchange
    31	    liquid), so the partition cannot ship on the spectral lane by merely lifting the gate. Either
    32	    the macmic sequential sub-cycle is ported to the spectral combined path (apply each module's
    33	    increment to the spectral state between modules -- extra spectral<->grid transforms per
    34	    sub-step, and the WB gradient's memory arena is ALREADY 643 GiB on one GPU before this), or
    35	    the partition stays refused on this lane and the WB deck documents it as an open departure.
    36	C3. Scope: nothing about the MPAS/hydrostatic lanes changes; the gate at combined.py:281 keeps its
    37	    meaning; (2) and (3) are already consistent.
    38	
    39	## Questions
    40	1. Is C2's ordering requirement really binding on the spectral lane, or is there a cheaper
    41	   faithful ordering (e.g. evaluating turbulence first and feeding its updated q_c only to the
    42	   microphysics within the existing parallel accumulate)?
    43	2. What is the memory and compile cost of a sequential sub-cycle inside a rollout step that is
    44	   already remat'd with nothing_saveable, at T63/L32?
    45	3. Does piece A alone (with the partition still refused) have any value, or is it dead code?
    46	4. What must be MEASURED before any of this lands?
    47	Verdict SHIP/HOLD first, then numbered findings with file:line.
rg: src/legoesm/atmosphere: No such file or directory (os error 2)
config/amip/amip_production.yaml:106:cld_macmic_num_steps: 3     # CAM6 namelist_defaults_cam.xml:1254 (mg2 + clubb_sgs,
config/amip/amip_production.yaml:137:hard_saturation_adjustment: false  # CAM6 clubb_do_liqsupersat = .false.
config/amip/amip_production.yaml:142:                                   # _mpas_hard_saturation_poststep), which is
config/amip/amip_production.yaml:151:hard_sat_ice_curve: false          # the ice-curve blend of that same
config/amip/amip_production.yaml:176:clubb_liquid_partition: true       # CAM6 clubb_intr: the closure's diagnosed
config/wb/campaign/spectral_t63_cam6.yaml:285:      hard_saturation_adjustment: false

exec
/usr/bin/bash -lc "nl -ba packages/atmosphere/legoesm/atmosphere/physics/combined.py | sed -n '220,305p;600,790p;1380,1600p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py | sed -n '520,568p;785,870p;920,1050p'" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 369ms:
   220	    # a silently-inert configuration (the 2026-07-23 --slab-land-active lesson).
   221	    if model_type != "mpas" and (f_land is not None or land_beta != 1.0):
   222	        raise ValueError(
   223	            "f_land/land_beta are MPAS-only land surface boundary knobs "
   224	            f"(model_type={model_type!r} would silently ignore them)."
   225	        )
   226	    if (sfc_albedo_override is not None or sfc_emissivity_override is not None) \
   227	            and model_type != "spectral_pe":
   228	        raise ValueError(
   229	            "sfc_albedo_override / sfc_emissivity_override are only wired for "
   230	            f"model_type='spectral_pe', got {model_type!r}."
   231	        )
   232	    if physics_cadence not in ("off", "write"):
   233	        raise ValueError(
   234	            f"physics_cadence must be 'off' or 'write'; got {physics_cadence!r}")
   235	    if physics_cadence != "off" and model_type != "mpas":
   236	        raise ValueError(
   237	            "physics_cadence (physics_update_steps > 1) is wired into the MPAS "
   238	            f"lane only; model_type={model_type!r} would silently ignore it")
   239	    if (not isinstance(physics_cadence_steps, int)
   240	            or isinstance(physics_cadence_steps, bool) or physics_cadence_steps < 1):
   241	        raise ValueError(
   242	            f"physics_cadence_steps must be an int >= 1, got {physics_cadence_steps!r}")
   243	    if physics_cadence == "off" and physics_cadence_steps != 1:
   244	        raise ValueError(
   245	            f"physics_cadence_steps={physics_cadence_steps} with physics_cadence="
   246	            "'off' would be silently ignored; pass physics_cadence='write'")
   247	    if (not isinstance(cld_macmic_num_steps, int)
   248	            or isinstance(cld_macmic_num_steps, bool) or cld_macmic_num_steps < 1):
   249	        raise ValueError(
   250	            f"cld_macmic_num_steps must be an int >= 1, got {cld_macmic_num_steps!r}")
   251	    if cld_macmic_num_steps != 1 and model_type not in ("hydrostatic", "mpas"):
   252	        raise ValueError(
   253	            "cld_macmic_num_steps > 1 (CAM macrophysics/microphysics sub-cycle) is "
   254	            f"wired into the hydrostatic/MPAS combined physics only; got {model_type!r}")
   255	    # CLUBB's liquid exchange REPLACES the host's cloud water, so the
   256	    # microphysics has to read the replaced value.  That ordering exists only in
   257	    # the macmic branch of ``_accumulate_step`` below: at N=1 it returns the
   258	    # PARALLEL ``_accumulate``, where every module is evaluated on the same
   259	    # start-of-step state and the tendencies are summed, leaving the
   260	    # microphysical sinks computed from the pre-exchange liquid.  Autoconversion
   261	    # goes as roughly the 2.5th power of cloud water, so that sink is wrong by
   262	    # nearly an order of magnitude -- and water is still conserved, so nothing
   263	    # fails.  Checked HERE, against the RESOLVED turbulence config, because the
   264	    # experiment-level flag is not the only way in: an authoritative
   265	    # turbulence_override carrying CLUBBConfig(liquid_partition=True), or a
   266	    # direct make_physics call, both reach this factory without it (codex).
   267	    # The MICROPHYSICS half alone deletes the model's only liquid source, so it
   268	    # is the more dangerous half to reach by itself -- the mirror of the
   269	    # turbulence-side guard below, and the same defect class both reviewers
   270	    # found on that side.  A direct ``make_physics`` call bypasses every
   271	    # ExperimentConfig guard, and they all key off the turbulence side.
   272	    if (_micro_liquid_from_closure_on(config.microphysics)
   273	            and not _clubb_liquid_partition_on(config.turbulence)):
   274	        raise ValueError(
   275	            "MorrisonConfig.liquid_from_closure=True without the CLUBB liquid "
   276	            "partition delivering liquid: the microphysics would stop "
   277	            "condensing and nothing would replace it, leaving the model with "
   278	            "NO cloud-liquid source at all. Enable both halves together "
   279	            "(ExperimentConfig.clubb_liquid_partition drives them), or "
   280	            "neither.")
   281	    if _clubb_liquid_partition_on(config.turbulence) and cld_macmic_num_steps < 2:
   282	        raise ValueError(
   283	            "CLUBBConfig.liquid_partition needs cld_macmic_num_steps>=2: the "
   284	            "closure REPLACES the host cloud water, so the microphysics must "
   285	            "run on the replaced value, and only the macro/micro sub-cycle "
   286	            f"applies the modules in sequence. At cld_macmic_num_steps="
   287	            f"{cld_macmic_num_steps} they are evaluated in parallel on the same "
   288	            "state and summed, leaving the microphysical sinks evaluated on the "
   289	            "pre-exchange liquid.")
   290	    if model_type == "hydrostatic":
   291	        fn = _make_hydrostatic_combined(
   292	            config, dt, column_mesh=column_mesh, need_rad=need_rad,
   293	            cld_macmic_num_steps=cld_macmic_num_steps)
   294	    elif model_type == "nonhydrostatic":
   295	        fn = _make_nonhydrostatic_combined(config, dt)
   296	    elif model_type == "spectral_pe":
   297	        fn = _make_spectral_pe_combined(
   298	            config, dt,
   299	            sfc_albedo_override=sfc_albedo_override,
   300	            sfc_emissivity_override=sfc_emissivity_override,
   301	        )
   302	    elif model_type == "mpas":
   303	        # MPAS (Voronoi mesh) uses the unified hydrostatic combined path.
   304	        fn = _make_hydrostatic_combined(
   305	            config, dt, model_type="mpas", column_mesh=column_mesh,
   600	        return out[0] if type(out) is tuple else out
   601	    shapes = jax.eval_shape(
   602	        _f, state, phys_state._replace(held_physics=_SEEDING), forcing)
   603	    return jax.tree_util.tree_map(lambda x: jnp.zeros(x.shape, x.dtype), shapes)
   604	
   605	
   606	def _make_hydrostatic_combined(config: PhysicsConfig, dt: float,
   607	                               model_type: str = "hydrostatic",
   608	                               column_mesh=None,
   609	                               need_rad: bool = True,
   610	                               f_land=None,
   611	                               land_beta: float = 1.0,
   612	                               budget_ledger: bool = False,
   613	                               budget_ledger_level_weight=None,
   614	                               physics_cadence: str = "off",
   615	                               physics_cadence_steps: int = 1,
   616	                               cld_macmic_num_steps: int = 1) -> Callable:
   617	    """Combined physics for any hydrostatic model (cubed-sphere, lat-lon, MPAS).
   618	
   619	    ``cld_macmic_num_steps`` (CAM6 namelist ``cld_macmic_num_steps``, 3 for
   620	    CLUBB+MG2 at dtime=1800): 1 (default) is the byte-identical parallel
   621	    split of every module on the pre-physics state.  N > 1 ports the CAM
   622	    ``tphysbc`` macro/micro loop: deep convection is applied to an
   623	    intermediate state first (CAM ``physics_update`` after
   624	    ``convect_deep_tend``), then turbulence (macrophysics) and microphysics
   625	    run N times sequentially at ``dt/N`` on that state, each sub-step's
   626	    increment applied before the next (``cld_macmic_ztodt``); their
   627	    tendencies, precipitation and ledger rows are averaged over the window
   628	    (CAM's ``1/N`` scaling).  Departures from CAM: radiation and gravity-wave
   629	    drag see the PRE-physics state (CAM: post-macmic / post-coupler) exactly
   630	    as the N=1 parallel split already does, and the whole convective
   631	    increment (including detrained condensate) is applied before sub-step 1
   632	    rather than spread over the sub-steps (CAM hands ``dlf`` per sub-step).
   633	
   634	    ``physics_cadence``: ``"off"`` (default, byte-identical to before);
   635	    ``"write"`` additionally stores the full tendency pytree of every call in
   636	    ``PhysicsState.held_physics`` (seed it with :func:`physics_cache_seed`)
   637	    and returns the state tendencies times ``physics_cadence_steps`` so the
   638	    dycore applies the whole physics increment on this step;
   639	    :func:`held_physics_variant` derives the step that applies zero state
   640	    tendency and re-publishes the cached diagnostic rates in between.
   641	
   642	    Uses the unified ``HydrostaticTendencies`` with optional ``dv_dt``.
   643	    When *model_type* is ``"mpas"``, the radiation factory is called
   644	    with ``"mpas"`` so that lat/lon extraction uses mesh.latCell/lonCell.
   645	
   646	    Issue #273 follow-up: when ``column_mesh`` is supplied, the
   647	    per-column radiation kernel runs sharded across the mesh.
   648	    """
   649	    # Static Python gate (a closure constant, never a traced value): False
   650	    # keeps every ledger branch out of the graph entirely, so the default path
   651	    # is byte-identical.  This is the documented feature-gating exception --
   652	    # jnp.where would trace BOTH branches.
   653	    _budget_ledger = bool(budget_ledger)
   654	    # Optional (nlev,) per-layer weight restricting every ledger row to a
   655	    # vertical band.  The column ledger cannot see a vertical-REDISTRIBUTION
   656	    # bias: convection's column water row is exactly zero by construction, so a
   657	    # scheme that moves water from the boundary layer into the mid-troposphere
   658	    # is invisible in it.  None = full column, byte-identical.
   659	    _ledger_level_weight = budget_ledger_level_weight
   660	    tagged_fns = []
   661	    # Aerosol-CCN specified-Nc coupling: the microphysics factory self-
   662	    # detects the switch from its own sub-config, but the radiation factory
   663	    # has no microphysics config, so the flag is threaded explicitly so its
   664	    # cloud-optics droplet number (Twomey r_eff) matches the microphysics
   665	    # fill.  False (default) keeps both paths byte-identical.
   666	    _nc_from_aerosol = _aerosol_ccn_active(config)
   667	    _activation_cfg = _aerosol_activation_config(config)
   668	    # CLUBB sub-grid cloud fraction -> radiation routing (marine-Sc albedo lever).
   669	    # A moist higher-order closure diagnoses a less-overcast cloud fraction than
   670	    # the RH grid-scale scheme; route it to the cloud optics when the user opts in
   671	    # AND a cf-producing closure is active.  ONLY diagnostic CLUBB
   672	    # (turbulence.scheme='clubb', CLUBBConfig.prognostic=False) writes the
   673	    # ``PhysicsState.cloud_fraction`` carry today (clubb_lite dropped its PDF
   674	    # moments; prognostic CLUBB carries packed moments, not a diagnosed cf).
   675	    # Requiring the producer keeps the radiation override off the zero-init carry
   676	    # (which would spuriously clear clouds).  Misconfiguration is LOUD, never a
   677	    # silent no-op (dispatch-hardening).
   678	    # BOTH CLUBB paths now publish a PDF cloud fraction.  The prognostic one
   679	    # used to be refused here on the grounds that it emitted "packed moments, no
   680	    # diagnosed cloud fraction" -- true of the OUTPUT, never of the closure: the
   681	    # moment advance computes the post-advance PDF cloud fraction and simply did
   682	    # not hand it out.  It does now, so the prognostic path is the BETTER source,
   683	    # not an unsupported one: its variance is carried state rather than a
   684	    # mixing-length estimate re-derived each step.
   685	    _turb_produces_cf = config.turbulence.scheme == "clubb"
   686	    _n_macmic = int(cld_macmic_num_steps)
   687	    if _n_macmic > 1 and (config.turbulence.scheme == "none"
   688	                          and config.microphysics.scheme == "none"):
   689	        raise ValueError(
   690	            f"cld_macmic_num_steps={_n_macmic} sub-cycles turbulence and "
   691	            "microphysics, but both are 'none'; set it to 1")
   692	    # Sub-cycled modules are BUILT with the sub-step length: implicit
   693	    # diffusion, CLUBB's moment advance and Morrison's process/sedimentation
   694	    # integration all take their own ``dt``.
   695	    _dt_sub = dt / _n_macmic
   696	    if config.radiation.use_clubb_cloud_fraction and not _turb_produces_cf:
   697	        raise ValueError(
   698	            "RadiationConfig.use_clubb_cloud_fraction=True requires a "
   699	            "cloud-fraction-producing turbulence closure "
   700	            "(turbulence.scheme='clubb', diagnostic or prognostic); got "
   701	            f"turbulence.scheme={config.turbulence.scheme!r}."
   702	        )
   703	    _use_clubb_cf = config.radiation.use_clubb_cloud_fraction
   704	    if config.radiation.scheme != "none":
   705	        tagged_fns.append((
   706	            make_radiation_physics(
   707	                config.radiation, model_type, column_mesh=column_mesh,
   708	                nc_from_aerosol=_nc_from_aerosol,
   709	                activation_config=_activation_cfg,
   710	                use_clubb_cloud_fraction=_use_clubb_cf,
   711	            ),
   712	            False,
   713	            None,
   714	        ))
   715	    if config.convection.scheme != "none":
   716	        tagged_fns.append((make_convection_physics(config.convection, model_type, dt), True, "conv_prog_profile"))
   717	    if config.turbulence.scheme != "none":
   718	        # MYNN-2.5 writes its prognostic ``qke = 2·TKE`` into a
   719	        # dedicated PhysicsState field so a restart-time scheme switch
   720	        # cannot silently feed the wrong moment as energy (Phase C
   721	        # codex iter-1 medium finding).
   722	        # qke (MYNN-2.5) / clubb_moments (prognostic CLUBB) / tke — must match the
   723	        # slot the per-model physics_fn reads (turbulence_carry_field, the
   724	        # config-aware refinement of turbulence_scheme_traits.energy_field).
   725	        _turb_sn, _, _turb_sc = get_turbulence_fn(config.turbulence)
   726	        _turb_field = turbulence_carry_field(_turb_sn, _turb_sc)
   727	        tagged_fns.append((
   728	            make_turbulence_physics(config.turbulence, model_type, _dt_sub,
   729	                                    f_land=f_land, land_beta=land_beta),
   730	            True,
   731	            _turb_field,
   732	        ))
   733	    if config.microphysics.scheme != "none":
   734	        tagged_fns.append((
   735	            make_microphysics_physics(config.microphysics, model_type, _dt_sub),
   736	            False, None))
   737	    if config.gravity_wave_drag.scheme != "none":
   738	        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, model_type, dt), True, "gwd_spectrum"))
   739	    # Macmic group of each module, PARALLEL to ``tagged_fns`` (same append
   740	    # order): "pre" = applied to the intermediate state before the loop
   741	    # (deep convection), "sub" = sub-cycled (turbulence, microphysics),
   742	    # "once" = evaluated once on the pre-physics state (radiation, GWD).
   743	    _macmic_group = []
   744	    if config.radiation.scheme != "none":
   745	        _macmic_group.append("once")
   746	    if config.convection.scheme != "none":
   747	        _macmic_group.append("pre")
   748	    if config.turbulence.scheme != "none":
   749	        _macmic_group.append("sub")
   750	    if config.microphysics.scheme != "none":
   751	        _macmic_group.append("sub")
   752	    if config.gravity_wave_drag.scheme != "none":
   753	        _macmic_group.append("once")
   754	    if len(_macmic_group) != len(tagged_fns):
   755	        raise AssertionError(
   756	            f"macmic group map has {len(_macmic_group)} entries for "
   757	            f"{len(tagged_fns)} physics modules — the two append sequences "
   758	            "have drifted; every module needs exactly one group.")
   759	
   760	    # Radiation sub-cycle plumbing.  Radiation is always ``tagged_fns[0]``
   761	    # when configured (appended first above), so in the accumulator below
   762	    # ``first.dT_dt`` IS the radiative heating contribution — that is what
   763	    # the full-physics variant caches into ``PhysicsState.rad_heating`` and
   764	    # the held-radiation variant (``need_rad=False``) re-uses.
   765	    # ``_non_rad_fns`` is the module list with radiation removed.
   766	    _has_rad = config.radiation.scheme != "none"
   767	    _non_rad_fns = tagged_fns[1:] if _has_rad else tagged_fns
   768	    _non_rad_groups = _macmic_group[1:] if _has_rad else _macmic_group
   769	
   770	    # --- Per-process budget-ledger row map (#1311 MPAS attribution) --------
   771	    # A list PARALLEL to ``tagged_fns`` giving each module's ledger row.  Kept
   772	    # parallel rather than widening the (fn, accepts_ps, field_name) tuple:
   773	    # that tuple is unpacked in three lifecycle-hook loops and in two OTHER
   774	    # model-type factories, none of which need the row.
   775	    # GWD maps to ``other_physics``, matching the FV pipeline, where GWD is
   776	    # part of the other_physics residual.
   777	    _ledger_row_of = []
   778	    if _budget_ledger:
   779	        from legoesm.diagnostics.process_ledger import (
   780	            ROW_CONVECTION, ROW_MICROPHYSICS, ROW_OTHER, ROW_RADIATION,
   781	            ROW_TURBULENCE,
   782	        )
   783	        if config.radiation.scheme != "none":
   784	            _ledger_row_of.append(ROW_RADIATION)
   785	        if config.convection.scheme != "none":
   786	            _ledger_row_of.append(ROW_CONVECTION)
   787	        if config.turbulence.scheme != "none":
   788	            _ledger_row_of.append(ROW_TURBULENCE)
   789	        if config.microphysics.scheme != "none":
   790	            _ledger_row_of.append(ROW_MICROPHYSICS)
  1380	
  1381	# ======================================================================
  1382	# Spectral PE
  1383	# ======================================================================
  1384	
  1385	def _make_spectral_pe_combined(
  1386	    config: PhysicsConfig, dt: float,
  1387	    sfc_albedo_override=None, sfc_emissivity_override=None,
  1388	) -> Callable:
  1389	    tagged_fns = []
  1390	    # Same producer gate as the hydrostatic combined: routing the CLUBB
  1391	    # cloud fraction to radiation needs a closure that writes it, else the
  1392	    # override would read the zero-initialised carry and clear every cloud.
  1393	    if (config.radiation.use_clubb_cloud_fraction
  1394	            and config.turbulence.scheme != "clubb"):
  1395	        raise ValueError(
  1396	            "RadiationConfig.use_clubb_cloud_fraction=True requires a "
  1397	            "cloud-fraction-producing turbulence closure "
  1398	            "(turbulence.scheme='clubb', diagnostic or prognostic); got "
  1399	            f"turbulence.scheme={config.turbulence.scheme!r}."
  1400	        )
  1401	    if config.radiation.scheme != "none":
  1402	        # The spectral radiation builder has a READ side for the CLUBB
  1403	        # cloud-fraction carry (and the CAM6 deepcu carries); the flag
  1404	        # selects it, and the dispatcher below forwards ``phys_state`` to
  1405	        # the fn that advertises ``_wants_phys_state_ro``.
  1406	        tagged_fns.append((make_radiation_physics(
  1407	            config.radiation, "spectral_pe",
  1408	            sfc_albedo_override=sfc_albedo_override,
  1409	            sfc_emissivity_override=sfc_emissivity_override,
  1410	            use_clubb_cloud_fraction=config.radiation.use_clubb_cloud_fraction,
  1411	        ), False, None))
  1412	    if config.convection.scheme != "none":
  1413	        tagged_fns.append((make_convection_physics(config.convection, "spectral_pe", dt), True, "conv_prog_profile"))
  1414	    if config.turbulence.scheme != "none":
  1415	        # Phase C codex iter-2 high: route MYNN-2.5 to ``qke`` on the
  1416	        # spectral PE combined path too.
  1417	        # qke (MYNN-2.5) / clubb_moments (prognostic CLUBB) / tke — must match the
  1418	        # slot the per-model physics_fn reads (turbulence_carry_field).
  1419	        _turb_sn, _, _turb_sc = get_turbulence_fn(config.turbulence)
  1420	        _turb_field = turbulence_carry_field(_turb_sn, _turb_sc)
  1421	        tagged_fns.append((
  1422	            make_turbulence_physics(config.turbulence, "spectral_pe", dt),
  1423	            True,
  1424	            _turb_field,
  1425	        ))
  1426	    if config.microphysics.scheme != "none":
  1427	        tagged_fns.append((make_microphysics_physics(config.microphysics, "spectral_pe", dt), False, None))
  1428	    if config.gravity_wave_drag.scheme != "none":
  1429	        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "spectral_pe", dt), True, "gwd_spectrum"))
  1430	
  1431	    def physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
  1432	        if not tagged_fns:
  1433	            zero_3d = jnp.zeros_like(state.vor_hat.data)
  1434	            zero_2d = jnp.zeros_like(state.lnps_hat.data)
  1435	            # Preserve the input state's tracer pytree structure as a
  1436	            # zero tendency so downstream tree.map(state, tendency)
  1437	            # works.  ``zero_like_tracers`` duck-types Field vs raw-array.
  1438	            zero_tracers = zero_like_tracers(state.tracers)
  1439	            zero_tend = SpectralHydrostaticState(
  1440	                vor_hat=state.vor_hat.replace(data=zero_3d),
  1441	                div_hat=state.div_hat.replace(data=zero_3d),
  1442	                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
  1443	                lnps_hat=state.lnps_hat.replace(data=zero_2d),
  1444	                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
  1445	                tracers=zero_tracers,
  1446	            )
  1447	            return zero_tend, None
  1448	
  1449	        # Compute grid-space diagnostics once and reuse across all active modules.
  1450	        shared_fields = spectral_pe_to_grid(state, grid, sigma_coord)
  1451	        phys_updates = {}
  1452	
  1453	        fn0, accepts_ps, field_name = tagged_fns[0]
  1454	        _fwd0 = ({"forcing": forcing}
  1455	                 if getattr(fn0, "_wants_forcing", False) else {})
  1456	        if accepts_ps:
  1457	            first, field_val = fn0(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state, **_fwd0)
  1458	            if field_val is not None and field_name is not None:
  1459	                # Multi-field updates (e.g., Bechtold's conv_prog_profile
  1460	                # and conv_stoch_state) are returned as a dict, which
  1461	                # we merge into ``phys_updates``.
  1462	                if isinstance(field_val, dict):
  1463	                    phys_updates.update(field_val)
  1464	                else:
  1465	                    phys_updates[field_name] = field_val
  1466	        else:
  1467	            if getattr(fn0, "_wants_phys_state_ro", False):
  1468	                # Read-only phys_state consumer (radiation reading the CLUBB
  1469	                # cloud-fraction / CAM6 deepcu carries): forward it but keep
  1470	                # the single-return contract -- no carry is written back.
  1471	                _fwd0 = {**_fwd0, "phys_state": phys_state}
  1472	            first = fn0(state, grid, sigma_coord, grid_fields=shared_fields, **_fwd0)
  1473	        vor_hat = first.vor_hat.data
  1474	        div_hat = first.div_hat.data
  1475	        T_hat = first.T_hat.data
  1476	        lnps_hat = first.lnps_hat.data
  1477	        phis_hat = first.phis_hat.data
  1478	
  1479	        # Tracer accumulation: start from the first module's
  1480	        # contribution if it has any, otherwise None.  Subsequent
  1481	        # modules are summed in the loop below.  Each value is kept
  1482	        # as a raw grid-space jax array (dropping the Field container)
  1483	        # so the per-key sum is a plain ``+``.  We re-wrap with the
  1484	        # state's container at the end for pytree-leaf consistency.
  1485	        def _grid_data(value):
  1486	            return value.data if hasattr(value, "data") else value
  1487	        accumulated_tracers = None
  1488	        if first.tracers is not None:
  1489	            accumulated_tracers = {
  1490	                k: _grid_data(v) for k, v in first.tracers.items()
  1491	            }
  1492	
  1493	        for fn, accepts_ps, field_name in tagged_fns[1:]:
  1494	            _fwd = ({"forcing": forcing}
  1495	                    if getattr(fn, "_wants_forcing", False) else {})
  1496	            if accepts_ps:
  1497	                t, field_val = fn(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state, **_fwd)
  1498	                if field_val is not None and field_name is not None:
  1499	                    # Match the first-iteration branch: dict updates
  1500	                    # (e.g., Bechtold's conv_prog_profile +
  1501	                    # conv_stoch_state + prng_key) must MERGE into
  1502	                    # phys_updates, not be assigned as a single
  1503	                    # blob under field_name.
  1504	                    if isinstance(field_val, dict):
  1505	                        phys_updates.update(field_val)
  1506	                    else:
  1507	                        phys_updates[field_name] = field_val
  1508	            else:
  1509	                if getattr(fn, "_wants_phys_state_ro", False):
  1510	                    _fwd = {**_fwd, "phys_state": phys_state}
  1511	                t = fn(state, grid, sigma_coord, grid_fields=shared_fields, **_fwd)
  1512	            vor_hat = vor_hat + t.vor_hat.data
  1513	            div_hat = div_hat + t.div_hat.data
  1514	            T_hat = T_hat + t.T_hat.data
  1515	            lnps_hat = lnps_hat + t.lnps_hat.data
  1516	            phis_hat = phis_hat + t.phis_hat.data
  1517	            # Per-tracer accumulation across modules.
  1518	            if t.tracers is not None:
  1519	                if accumulated_tracers is None:
  1520	                    accumulated_tracers = {
  1521	                        k: _grid_data(v) for k, v in t.tracers.items()
  1522	                    }
  1523	                else:
  1524	                    for k, v in t.tracers.items():
  1525	                        v_data = _grid_data(v)
  1526	                        if k in accumulated_tracers:
  1527	                            accumulated_tracers[k] = (
  1528	                                accumulated_tracers[k] + v_data
  1529	                            )
  1530	                        else:
  1531	                            accumulated_tracers[k] = v_data
  1532	
  1533	        # Build the final tracers dict for the combined tendency,
  1534	        # mirroring the input state's container types so pytree leaves
  1535	        # match downstream tree.map(state, tendency) calls.  When no
  1536	        # physics module touched tracers we emit ``zero_like_tracers``
  1537	        # of the input state's tracers (the original safe default).
  1538	        if state.tracers is None:
  1539	            tracers_combined = None
  1540	        elif accumulated_tracers is None:
  1541	            tracers_combined = zero_like_tracers(state.tracers)
  1542	        else:
  1543	            tracers_combined = {}
  1544	            for k, template in state.tracers.items():
  1545	                if k in accumulated_tracers:
  1546	                    arr = accumulated_tracers[k]
  1547	                else:
  1548	                    arr = (
  1549	                        jnp.zeros_like(template.data)
  1550	                        if hasattr(template, "data")
  1551	                        else jnp.zeros_like(template)
  1552	                    )
  1553	                if hasattr(template, "data") and hasattr(template, "replace"):
  1554	                    tracers_combined[k] = template.replace(data=arr)
  1555	                else:
  1556	                    tracers_combined[k] = arr
  1557	        combined = SpectralHydrostaticState(
  1558	            vor_hat=first.vor_hat.replace(data=vor_hat),
  1559	            div_hat=first.div_hat.replace(data=div_hat),
  1560	            T_hat=first.T_hat.replace(data=T_hat),
  1561	            lnps_hat=first.lnps_hat.replace(data=lnps_hat),
  1562	            phis_hat=first.phis_hat.replace(data=phis_hat),
  1563	            tracers=tracers_combined,
  1564	        )
  1565	        phys_state_out = update_physics_state(phys_state, phys_updates)
  1566	        return combined, phys_state_out
  1567	
  1568	    physics_fn = _attach_lifecycle_hooks(physics_fn, tagged_fns)
  1569	    # Marker: the combined fn consumes a per-step traced ``forcing``
  1570	    # dict when any sub-physics advertises ``_wants_forcing`` (currently
  1571	    # the spectral_pe radiation factory: T_sfc/o3_vmr/aerosol_od/ghg_vmr).
  1572	    physics_fn._wants_forcing = any(
  1573	        getattr(fn, "_wants_forcing", False) for fn, _, _ in tagged_fns
  1574	    )
  1575	    return physics_fn
   520	    -------
   521	    Callable
   522	        Physics function with the correct signature for the model.
   523	    """
   524	    if model_type != "mpas" and (f_land is not None or land_beta != 1.0):
   525	        raise ValueError(
   526	            "f_land/land_beta are the MPAS land surface boundary knobs; the "
   527	            f"{model_type!r} pipeline has its own land tile (they would be "
   528	            "silently inert here). Drop them or use model_type='mpas'."
   529	        )
   530	    if model_type != "mpas":
   531	        _sub = getattr(materialize_sub_config(turbulence_config),
   532	                       turbulence_config.scheme, None)
   533	        _srf = getattr(_sub, "surface", None)
   534	        # z_ref_model_level now reaches EVERY lane: each turbulence kernel
   535	        # calls surface_fluxes_at_lowest_level, which honours it from the
   536	        # config with the level height the kernel already has.  The saline
   537	        # ocean humidity still cannot: it applies to the OCEAN fraction only,
   538	        # and the structured-grid lanes deliberately have no land fraction in
   539	        # the turbulence factory (see the f_land guard above), so applying it
   540	        # would put sea water under the continents.  Raise rather than ignore.
   541	        # The liquid partition hands back a cloud-liquid tendency PAIRED with a
   542	        # vapour tendency that has had that liquid removed. Only the MPAS lane
   543	        # routes it to a tracer today; dropping it on a lane that cannot would
   544	        # destroy exactly that much water every step, silently. Refuse instead.
   545	        if _sub is not None and getattr(_sub, "liquid_partition", False):
   546	            raise NotImplementedError(
   547	                "CLUBBConfig.liquid_partition returns a cloud-liquid tendency "
   548	                f"(TurbulenceOutput.dq_c_dt) that the {model_type!r} turbulence "
   549	                "lane does not route to a condensate tracer, so the liquid the "
   550	                "closure removed from vapour would be destroyed. Use "
   551	                "model_type='mpas', or extend this lane to apply dq_c_dt.")
   552	        if _srf is not None and getattr(_srf, "ocean_q_sfc_saline", False):
   553	            raise NotImplementedError(
   554	                "SurfaceLayerConfig.ocean_q_sfc_saline needs a land fraction "
   555	                "to apply the sea-water humidity to the ocean tile only, and "
   556	                f"the {model_type!r} turbulence factory has none (f_land is "
   557	                "the MPAS bridge's knob). Pass surface=...\n"
   558	                "._replace(ocean_q_sfc_saline=False) on this lane, or extend "
   559	                "the lane to carry a land fraction.")
   560	    if model_type == "hydrostatic":
   561	        return _make_hydrostatic_turbulence(turbulence_config, dt)
   562	    elif model_type == "nonhydrostatic":
   563	        return _make_nonhydrostatic_turbulence(turbulence_config, dt)
   564	    elif model_type == "spectral_pe":
   565	        return _make_spectral_pe_turbulence(turbulence_config, dt)
   566	    elif model_type == "mpas":
   567	        # UNBLOCKED: ``MPASPrimitiveEquationModel.step`` now (a) carries
   568	        # ``state.tracers`` (q_v advected by the dycore — Phase B) and (b)
   785	    # raises with its own name on the first step instead of as a TypeError
   786	    # inside a traced column.
   787	    _accepts_surface_flux = kernel_accepts_surface_flux(turb_fn)
   788	    # Static feature gate (build-time closure constant, not traced): with it off
   789	    # the liquid exchange below is absent from the trace entirely.
   790	    _liquid_partition = bool(getattr(scheme_config, "liquid_partition", False))
   791	
   792	    def physics_fn(state, mesh, sigma_coord, phys_state=None, forcing=None):
   793	        from legoesm.grids.voronoi import (
   794	            cell_vector_to_edge_normal,
   795	            reconstruct_cell_velocity,
   796	        )
   797	
   798	        tke_out = None
   799	        if turb_fn is None:
   800	            # Build zero-tendency in MPAS layout (edge-centric u).
   801	            zero_edges = jnp.zeros_like(state.u.data)
   802	            zero_cells = jnp.zeros_like(state.T.data)
   803	            zero_ps = jnp.zeros_like(state.p_s.data)
   804	            tendencies = HydrostaticTendencies(
   805	                du_dt=state.u.replace(data=zero_edges),
   806	                dv_dt=None,
   807	                dT_dt=state.T.replace(data=zero_cells),
   808	                dp_s_dt=state.p_s.replace(data=zero_ps),
   809	                dphis_dt=state.phis.replace(data=zero_ps),
   810	            )
   811	            return tendencies, tke_out
   812	
   813	        u_edge = state.u.data       # (nEdges, nlev)
   814	        T = state.T.data            # (nCells, nlev)
   815	        p_s = state.p_s.data        # (nCells,)
   816	        nlev = sigma_coord.n_levels
   817	        nCells = T.shape[0]
   818	
   819	        # Edge → cell wind reconstruction (Perot 2000).
   820	        u_cell, v_cell = reconstruct_cell_velocity(u_edge, mesh)
   821	
   822	        # Pressures
   823	        p_full = sigma_coord.pressure_at_full(p_s)  # (nCells, nlev)
   824	        p_half = sigma_coord.pressure_at_half(p_s)  # (nCells, nlev+1)
   825	
   826	        # Column-format inputs (already 1D × nlev, so reshape is a no-op).
   827	        T_col = T.reshape(nCells, nlev)
   828	        u_col = u_cell.reshape(nCells, nlev)
   829	        v_col = v_cell.reshape(nCells, nlev)
   830	        p_full_col = p_full.reshape(nCells, nlev)
   831	        p_half_col = p_half.reshape(nCells, nlev + 1)
   832	
   833	        _state_dtype = T.dtype
   834	        if state.tracers is not None and "q_v" in state.tracers:
   835	            _qv_raw = state.tracers["q_v"]
   836	            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
   837	            q_v_col = _qv_data.reshape(nCells, nlev)
   838	        else:
   839	            q_v_col = jnp.zeros((nCells, nlev), dtype=_state_dtype)
   840	
   841	        # Host cloud liquid, for the CLUBB liquid partition only.  Absent the
   842	        # lever this stays None and nothing below it is traced.  The tracer must
   843	        # EXIST: with the lever on, a missing q_c would seed the closure with no
   844	        # liquid and then write its rcm back over a tracer nobody carries, which
   845	        # is the one-way ratchet the seeding exists to prevent.
   846	        _qc_col = None
   847	        if _liquid_partition:
   848	            # BOTH halves of the pair, not just the liquid: the vapour block
   849	            # above substitutes zeros for a missing 'q_v' and the tendency
   850	            # section below then emits no vapour tendency, so a state carrying
   851	            # liquid but no vapour would evaporate liquid into a tendency
   852	            # nothing applies -- water destroyed, silently (codex).
   853	            _missing = [n for n in ("q_v", "q_c")
   854	                        if state.tracers is None or n not in state.tracers]
   855	            if _missing:
   856	                raise ValueError(
   857	                    "CLUBBConfig.liquid_partition exchanges water between the "
   858	                    f"vapour and liquid tracers, and {_missing} is not carried "
   859	                    "by this state. Both must exist, or the half that is "
   860	                    "missing is destroyed. Carry them, or switch the partition "
   861	                    "off.")
   862	            _qc_raw = state.tracers["q_c"]
   863	            _qc_col = (_qc_raw.data if hasattr(_qc_raw, "data")
   864	                       else _qc_raw).reshape(nCells, nlev)
   865	
   866	        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col)
   867	        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col)
   868	
   869	        # Surface temperature for the bulk fluxes: the prescribed SST from the
   870	        # per-step traced ``forcing["T_sfc"]`` (AMIP path) when supplied, so
   920	                     + _f_land_col * jnp.asarray(
   921	                         _qsfc_traced, dtype=q_sfc.dtype).reshape(nCells))
   922	        _beta_traced = (forcing.get("beta_land")
   923	                        if forcing is not None else None)
   924	        if _qsfc_traced is not None:
   925	            pass                     # the solved humidity supersedes beta
   926	        elif _beta_traced is not None:
   927	            if f_land is None:
   928	                raise ValueError(
   929	                    "forcing['beta_land'] (traced per-cell beta_soil) "
   930	                    "requires the land fraction to be threaded into the "
   931	                    "turbulence factory (make_physics f_land=...); the "
   932	                    "driver must pass f_land whenever mpas_land_beta_soil "
   933	                    "is enabled."
   934	                )
   935	            from legoesm.atmosphere.physics.turbulence.surface_layer import (
   936	                beta_limited_surface_humidity,
   937	            )
   938	            _f_land_col = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
   939	            q_sfc = beta_limited_surface_humidity(
   940	                q_sfc, q_v_col[:, -1], _f_land_col,
   941	                jnp.asarray(_beta_traced, dtype=q_sfc.dtype).reshape(nCells))
   942	        elif f_land is not None and land_beta != 1.0:
   943	            from legoesm.atmosphere.physics.turbulence.surface_layer import (
   944	                beta_limited_surface_humidity,
   945	            )
   946	            _f_land_col = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
   947	            q_sfc = beta_limited_surface_humidity(
   948	                q_sfc, q_v_col[:, -1], _f_land_col, land_beta)
   949	        if getattr(_surf, "ocean_q_sfc_saline", False):
   950	            # Sea water at the SURFACE pressure (the coupled lane's convention)
   951	            # for the ocean fraction: replace that fraction's fresh-water share
   952	            # of the blend; the land share is untouched.
   953	            from legoesm.core.bulk_flux import ocean_surface_q_sat
   954	            _q_ocean = ocean_surface_q_sat(
   955	                T_sfc, p_s.reshape(nCells), thermo_convention=_surf.thermo_convention,
   956	                bulk_scheme=_surf.bulk_scheme, saline_factor=_Q_SAT_SALINE_FACTOR)
   957	            _fo = (1.0 - jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
   958	                   if f_land is not None else 1.0)
   959	            q_sfc = q_sfc + _fo * (_q_ocean - _q_sfc_fresh)
   960	
   961	        # A per-step prescribed surface flux (a diurnal cycle the run-constant
   962	        # config scalar cannot carry) enters here; identity without one.
   963	        step_config = _resolve_prescribed_surface_fluxes(
   964	            scheme_config, phys_state, rho)
   965	        # ``forcing["shflx_land"]`` / ``forcing["lhflx_land"]`` — the land
   966	        # scheme's OWN turbulent fluxes, blended by land fraction into the
   967	        # surface flux the BL scheme consumes.  This exists because handing
   968	        # over the canopy's HUMIDITY was measured insufficient: the canopy
   969	        # solved its flux against ITS aerodynamic resistance, so its boundary
   970	        # humidity sits close to the air by construction, and the atmosphere
   971	        # re-applying its own resistance to that already-collapsed gradient
   972	        # delivered ~a tenth of the canopy's flux (Amazon latent heat 78 W/m2
   973	        # offline -> 7 coupled, land 10 K cold in 30 days).  A flux is what
   974	        # the land solved; a flux is what crosses the boundary.  Momentum and
   975	        # the ocean/ice fraction keep the scheme's own bulk computation.
   976	        _shf_land = (forcing.get("shflx_land") if forcing is not None else None)
   977	        _surface_flux = None
   978	        if _shf_land is not None:
   979	            if f_land is None:
   980	                raise ValueError(
   981	                    "forcing['shflx_land'] requires f_land in the turbulence "
   982	                    "factory (make_physics f_land=...).")
   983	            if not _accepts_surface_flux:
   984	                raise ValueError(
   985	                    f"turbulence scheme {scheme_name!r} cannot be handed the "
   986	                    "land surface fluxes: its kernel takes no 'surface_flux' "
   987	                    "argument, so it computes its own from T_sfc/q_sfc. "
   988	                    "Silently dropping them would run the advertised land "
   989	                    "coupling with a surface flux the land model never "
   990	                    f"solved. Use one of {schemes_accepting_surface_flux()}, "
   991	                    "or teach this scheme the argument.")
   992	            from legoesm.atmosphere.physics.turbulence.surface_layer import (
   993	                surface_fluxes_at_lowest_level,
   994	            )
   995	            _fl = jnp.asarray(f_land, dtype=q_sfc.dtype).reshape(nCells)
   996	            # Through the shared helper rather than an inline copy: this
   997	            # branch had its own adjustment, which kept the warmed air while
   998	            # the bulk law dropped the height on a constant-coefficient
   999	            # scheme -- the same one-sided contrast the helper exists to
  1000	            # prevent, surviving in the one place that did not call it
  1001	            # (codex).  Height measured from the LOCAL surface.
  1002	            _tx, _ty, _sh, _lh, _us = surface_fluxes_at_lowest_level(
  1003	                u_col[:, -1], v_col[:, -1], T_col[:, -1], q_v_col[:, -1],
  1004	                T_sfc, q_sfc, rho[:, -1], step_config.surface,
  1005	                z_full[:, -1] - z_half[:, -1],
  1006	            )
  1007	            _lh_land = jnp.asarray(
  1008	                forcing["lhflx_land"], dtype=q_sfc.dtype).reshape(nCells)
  1009	            _sh_land = jnp.asarray(_shf_land, dtype=q_sfc.dtype).reshape(nCells)
  1010	            _surface_flux = (
  1011	                _tx, _ty,
  1012	                (1.0 - _fl) * _sh + _fl * _sh_land,
  1013	                (1.0 - _fl) * _lh + _fl * _lh_land,
  1014	                _us,
  1015	            )
  1016	
  1017	        # Forwarded on BOTH branches: clubb and clubb_lite carry a prognostic
  1018	        # energy field and still accept the flux, so gating this on the carry
  1019	        # would drop the land coupling for exactly those two.
  1020	        _sfc_kw = ({"surface_flux": _surface_flux}
  1021	                   if _surface_flux is not None else {})
  1022	        if _liquid_partition:
  1023	            _sfc_kw = dict(_sfc_kw, q_c=_qc_col)
  1024	
  1025	        # Radiative-heating kwargs: prognostic CLUBB only, and only when the
  1026	        # host cached a rad_heating.  CACHE LAG: combined.py refreshes the
  1027	        # cache AFTER turbulence, so this is the PRECEDING radiation solve's
  1028	        # heating.  {} (no kwarg at all) for every other scheme and the
  1029	        # diagnostic CLUBB path — their turb_fn signatures do not accept it.
  1030	        _rad_kw = _prognostic_clubb_rad_kwargs(
  1031	            carry_field, phys_state, nCells, nlev, _state_dtype)
  1032	
  1033	        if needs_tke:
  1034	            tke_in = _read_turb_carry(
  1035	                phys_state, carry_field, nCells, nlev, scheme_config, _state_dtype)
  1036	            turb_out, tke_new = turb_fn(
  1037	                u_col, v_col, T_col, q_v_col, tke_in,
  1038	                p_full_col, p_half_col, z_full, z_half,
  1039	                T_sfc, q_sfc, rho, dt, step_config,
  1040	                **_sfc_kw, **_rad_kw,
  1041	            )
  1042	            tke_out = tke_new
  1043	        else:
  1044	            turb_out = turb_fn(
  1045	                u_col, v_col, T_col, q_v_col,
  1046	                p_full_col, p_half_col, z_full, z_half,
  1047	                T_sfc, q_sfc, rho, dt, step_config,
  1048	                **_sfc_kw,
  1049	            )
  1050	        if _liquid_partition and turb_out.dq_c_dt is None:

exec
/usr/bin/bash -lc "rg -n 'def _make_spectral|dq_c_dt|q_c=|liquid_partition' packages/atmosphere/legoesm/atmosphere/physics/turbulence/{integration,clubb}.py; rg -n '643|nothing_saveable|cld_macmic_num_steps|hard_sat_ice_curve|clubb_liquid_partition' packages/training scripts/train* scripts/wb docs/wb config/wb/campaign/spectral_t63_cam6.yaml 2>/dev/null | head -100; nl -ba packages/atmosphere/legoesm/atmosphere/physics/combined.py | sed -n '990,1125p;1400,1533p'" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 468ms:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:545:        if _sub is not None and getattr(_sub, "liquid_partition", False):
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:547:                "CLUBBConfig.liquid_partition returns a cloud-liquid tendency "
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:548:                f"(TurbulenceOutput.dq_c_dt) that the {model_type!r} turbulence "
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:551:                "model_type='mpas', or extend this lane to apply dq_c_dt.")
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:790:    _liquid_partition = bool(getattr(scheme_config, "liquid_partition", False))
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:847:        if _liquid_partition:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:857:                    "CLUBBConfig.liquid_partition exchanges water between the "
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1022:        if _liquid_partition:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1023:            _sfc_kw = dict(_sfc_kw, q_c=_qc_col)
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1050:        if _liquid_partition and turb_out.dq_c_dt is None:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1053:                "partition but returned no dq_c_dt, so the liquid removed from "
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1080:        if _liquid_partition:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1086:                    data=turb_out.dq_c_dt.reshape(_qc_raw.data.shape),
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1087:                    name="dq_c_dt_turb",
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1090:                tracer_tends["q_c"] = turb_out.dq_c_dt.reshape(_qc_raw.shape)
packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1301:def _make_spectral_pe_turbulence(
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:663:    liquid_partition : bool
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:702:        ``TurbulenceOutput.dq_c_dt``; lanes that cannot are refused at
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:740:    liquid_partition: bool = False
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:5499:    # grid mean.  ``liquid_partition`` consumes this key, not ``rcm``.
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6317:    when ``config.liquid_partition`` is set; see that field. Passing it with the
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6324:    liquid partition on, ``diagnostics["dq_c_dt"]`` carries the cloud-liquid
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6330:    if config.liquid_partition:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6333:                "clubb_step: liquid_partition=True requires the host cloud "
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6340:            "clubb_step: q_c was supplied but liquid_partition=False, so the "
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6351:    if config.liquid_partition:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6482:    if config.liquid_partition:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6493:        diags = dict(diags, dq_c_dt=(rcm_new - q_c) / dt)
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6641:    if q_c is not None and not config.liquid_partition:
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6644:            "liquid_partition=False, so the cloud liquid would be silently "
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6646:    _liq_kw = {"q_c": q_c} if config.liquid_partition else {}
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6656:        dq_c_dt = diags.get("dq_c_dt")
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6665:        _liq_on = config.liquid_partition
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6685:            ql_new = ql_c + dt_sub * diag["dq_c_dt"] if _liq_on else ql_c
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6697:        dq_c_dt = (ql_f - q_c) / dt if _liq_on else None
packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6736:        h_pbl=h_pbl, cloud_fraction=cloud_fraction_td, dq_c_dt=dq_c_dt)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md:10:  1. clubb_liquid_partition: true   (CLUBB's diagnosed liquid handed to the host cloud water)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md:12:  3. hard_sat_ice_curve: false
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md:19: - combined.py:251 refuses cld_macmic_num_steps>1 outside ("hydrostatic","mpas"); combined.py:281
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md:20:   requires cld_macmic_num_steps>=2 whenever the partition is on, because the closure REPLACES the
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md:34:    sub-step, and the WB gradient's memory arena is ALREADY 643 GiB on one GPU before this), or
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md:44:   already remat'd with nothing_saveable, at T63/L32?
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1177:  1. clubb_liquid_partition: true   (CLUBB's diagnosed liquid handed to the host cloud water)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1179:  3. hard_sat_ice_curve: false
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1186: - combined.py:251 refuses cld_macmic_num_steps>1 outside ("hydrostatic","mpas"); combined.py:281
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1187:   requires cld_macmic_num_steps>=2 whenever the partition is on, because the closure REPLACES the
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1201:    sub-step, and the WB gradient's memory arena is ALREADY 643 GiB on one GPU before this), or
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1211:   already remat'd with nothing_saveable, at T63/L32?
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1622:/usr/bin/bash -lc "rg --files -g AGENTS.md -g '*combined.py' -g '*integration.py' | head -70; git worktree list; nl -ba docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md; rg -n 'liquid_partition|cld_macmic_num_steps|hard_sat' config/wb/campaign/spectral_t63_cam6.yaml config/amip/amip_production.yaml src/legoesm/atmosphere" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1728:/work/bd1083/b309178/diffESM/legoesm_pg/wt_pr_res6                                                                                 643454dd4 [feat/amip-res6-default]
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1799:    10	  1. clubb_liquid_partition: true   (CLUBB's diagnosed liquid handed to the host cloud water)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1801:    12	  3. hard_sat_ice_curve: false
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1808:    19	 - combined.py:251 refuses cld_macmic_num_steps>1 outside ("hydrostatic","mpas"); combined.py:281
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1809:    20	   requires cld_macmic_num_steps>=2 whenever the partition is on, because the closure REPLACES the
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1823:    34	    sub-step, and the WB gradient's memory arena is ALREADY 643 GiB on one GPU before this), or
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1833:    44	   already remat'd with nothing_saveable, at T63/L32?
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1838:config/amip/amip_production.yaml:106:cld_macmic_num_steps: 3     # CAM6 namelist_defaults_cam.xml:1254 (mg2 + clubb_sgs,
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1841:config/amip/amip_production.yaml:151:hard_sat_ice_curve: false          # the ice-curve blend of that same
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1842:config/amip/amip_production.yaml:176:clubb_liquid_partition: true       # CAM6 clubb_intr: the closure's diagnosed
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1875:   247	    if (not isinstance(cld_macmic_num_steps, int)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1876:   248	            or isinstance(cld_macmic_num_steps, bool) or cld_macmic_num_steps < 1):
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1878:   250	            f"cld_macmic_num_steps must be an int >= 1, got {cld_macmic_num_steps!r}")
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1879:   251	    if cld_macmic_num_steps != 1 and model_type not in ("hydrostatic", "mpas"):
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1881:   253	            "cld_macmic_num_steps > 1 (CAM macrophysics/microphysics sub-cycle) is "
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1901:   273	            and not _clubb_liquid_partition_on(config.turbulence)):
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1907:   279	            "(ExperimentConfig.clubb_liquid_partition drives them), or "
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1909:   281	    if _clubb_liquid_partition_on(config.turbulence) and cld_macmic_num_steps < 2:
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1911:   283	            "CLUBBConfig.liquid_partition needs cld_macmic_num_steps>=2: the "
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1914:   286	            f"applies the modules in sequence. At cld_macmic_num_steps="
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1915:   287	            f"{cld_macmic_num_steps} they are evaluated in parallel on the same "
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1921:   293	            cld_macmic_num_steps=cld_macmic_num_steps)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1950:   616	                               cld_macmic_num_steps: int = 1) -> Callable:
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1953:   619	    ``cld_macmic_num_steps`` (CAM6 namelist ``cld_macmic_num_steps``, 3 for
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:1977:   643	    When *model_type* is ``"mpas"``, the radiation factory is called
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:2020:   686	    _n_macmic = int(cld_macmic_num_steps)
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:2024:   690	            f"cld_macmic_num_steps={_n_macmic} sub-cycles turbulence and "
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_codex.md:2589:/usr/bin/bash -lc "rg -n 'def _make_spectral|dq_c_dt|q_c=|liquid_partition' packages/atmosphere/legoesm/atmosphere/physics/turbulence/{integration,clubb}.py; rg -n '643|nothing_saveable|cld_macmic_num_steps|hard_sat_ice_curve|clubb_liquid_partition' packages/training scripts/train* scripts/wb docs/wb config/wb/campaign/spectral_t63_cam6.yaml 2>/dev/null | head -100; nl -ba packages/atmosphere/legoesm/atmosphere/physics/combined.py | sed -n '990,1125p;1400,1533p'" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
docs/wb/reviews/2026-09-24_cam6_wb_prompt.md:33:CLUBB+Morrison at 600 s (cld_macmic_num_steps=3) plus sedimentation CFL
docs/wb/reviews/2026-09-24_cam6_wb_prompt.md:36:cld_macmic_num_steps>1 unless model_type is "hydrostatic" or "mpas", and
docs/wb/reviews/2026-09-24_cam6_wb_glm.md:11:1. **C2 is worse than you claim** — the guard is real, but your deck never sends `cld_macmic_num_steps` at all, so nothing will even refuse. The failure is *silent*, not loud, and the spectral path doesn't thread the kwarg even if the guard were deleted.
docs/wb/reviews/2026-09-24_cam6_wb_glm.md:39:**VERIFIED-FROM-EXCERPT:** the guard `if cld_macmic_num_steps != 1 and model_type not in ("hydrostatic", "mpas"): raise` fires for `model_type="spectral_pe"`. And critically, the dispatch calls `_make_spectral_pe_combined(config, dt, sfc_albedo_override=..., sfc_emissivity_override=...)` **without `cld_macmic_num_steps`**. So this is not a conservative guard over implemented functionality — the sub-cycle is **not wired into the spectral path at all**. Deleting the guard buys you nothing; crossing this requires code in `_make_spectral_pe_combined`.
docs/wb/reviews/2026-09-24_cam6_wb_glm.md:41:**The hole in your claim:** your deck contains **no `cld_macmic_num_steps` row anywhere**. So the `ValueError` you read as "REFUSING" will never fire on this deck — nothing asks for it. Unless the WB builder forwards such a key (YOUR-READ, unverified), the run will proceed at 1800 s with **zero diagnostics**. Silent > loud as failure modes. Verify with: grep the WB builder for `cld_macmic_num_steps` / `make_physics(` call sites, and a smoke call `make_physics(cfg, model_type="spectral_pe", cld_macmic_num_steps=3)` to confirm the guard fires at all.
docs/wb/reviews/2026-09-24_cam6_wb_glm.md:43:**The reconstruction that makes C2 sharper (INFERENCE from your numbers — verify):** AMIP is dt=112.5, `physics_update_steps 16` → 112.5×16 = **1800 s physics**, with `cld_macmic_num_steps=3` → **600 s macro/micro**; `rad_update_steps 32` → 3600 s = 1 h radiation, matching CAM6. So the AMIP baseline is *already* a 1800 s physics call made safe by internal 600 s sub-cycling. Your deck is a 1800 s physics call **without** it — i.e., macro/micro at 3× the step length that was already demonstrated to cold-collapse to 51 K. You are not merely missing a tuning row; you are running a configuration strictly worse than the one that failed.
docs/wb/reviews/2026-09-24_cam6_wb_glm.md:47:**Cheapest measurement to settle it:** a **no-gradient forward roll**: initialize from a reanalysis snapshot interpolated to T63/L32 sigma, run 20–40 steps of dt=1800 with the deck's exact physics, watch column-min T and total condensate path. Minutes, single device, no data pipeline. A 1-column tropical warm profile variant is even cheaper. If min-T dives below ~180 K or condensate grows monotonically → collapse confirmed, task blocked pending the code change (thread `cld_macmic_num_steps` through `_make_spectral_pe_combined`, mirroring the hydrostatic path).
docs/wb/reviews/2026-09-24_cam6_wb_glm.md:87:2. **Stability (C2):** the no-grad 20-step roll. If it collapses (I expect it to), the fix is code — thread `cld_macmic_num_steps` through `_make_spectral_pe_combined` — not configuration, and that's a scoping conversation with whoever owns the spectral lane.
docs/wb/reviews/2026-09-24_cam6_wb_wiring_r3_prompt.md:15:                                 prevent_cse=True, policy=nothing_saveable)(phys0)
docs/wb/reviews/2026-09-24_cam6_wb_wiring_r3_prompt.md:46:    nothing_saveable. Under eqx.filter_value_and_grad, does it add one
docs/wb/reviews/2026-09-24_cam6_wb_wiring_reviews.md:64:- Gradient: `phys_state` rides the scan carry → `cloud_fraction_override` → cloud optics → RRTMGP, under `nothing_saveable` at the step level (recompute) and default policy at the radiation core. Backward through both checkpoints is supported; nothing here severs it. Same differentiability contract as hydrostatic. Note the (desirable, but name-it-under-F) consequence: **training gradients now couple radiation loss → CLUBB carry**, which no spectral baseline ever had (cam6 previously raised).
docs/wb/reviews/2026-09-24_cam6_wb_wiring_reviews.md:176:`nothing_saveable` means the checkpoint saves zero intermediates and **recomputes** the entry in backward — so no un-remat'd physics tape is added. Cost is one extra physics forward (trace time) plus one extra recompute per backward pass, outside the scan, i.e. O(1) versus n_steps in-scan entries. Symmetric with `step_fn_ckpt`'s own treatment. `prevent_cse=True` is the *safe* direction here (omitting it under remat is what causes the classic CSE/backward-differential artifacts); its only cost is disallowed sharing. One noteworthy improvement: the first window's cloud fraction previously had **zero gradient** w.r.t. CLUBB parameters (seed carry, no param dependence); the warm-up now threads a correct gradient path through it. That is a training-signal change — an improvement, but name it under F.
docs/wb/reviews/2026-09-24_cam6_wb_wiring_reviews.md:233:No debris found. Specifically: no unused imports in the test file; the 10-passing count reconciles exactly with the file's contents (9 functions, one parametrized ×2) — nothing silently missing from the round-3 truncation; the three gate sites (combined, split factory, radiation builder) use consistent messages and the tests match on the shared phrase; the `_wants_phys_state_ro` re-export on the split `rad_fn` wrapper covers the seam the rollout actually keys on; `_call_rad` omits the kwarg when `phys_state is None`, so markerless/stateless arms bind old signatures; the warm-up uses the same `_ps_entry` the per-step harvest uses, under the same `nothing_saveable` remat — self-consistent by construction. Non-blocking observations, none new in this round: (i) the combined producer gate fires even when `radiation.scheme == "none"` (flag moot) — strictness matching the hydrostatic gate; (ii) flag-on + non-cam6 cloud scheme with no forwarded carry degrades silently to the RH path (only cam6 refuses loudly) — same convention as hydrostatic; (iii) the owner-decision date appears twice in one docstring — cosmetic.
   990	        return (du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
   991	                combined_tracer_tends, phys_updates, first, precip_accum,
   992	                sfc_diag_extras, _led)
   993	
   994	    def _build_combined(first, du_dt, dv_dt, dT_dt, dp_s_dt, dphis_dt,
   995	                        combined_tracer_tends):
   996	        # Build tracer_tendencies dict with Field wrappers
   997	        tracer_tends_out = None
   998	        if combined_tracer_tends:
   999	            tracer_tends_out = {
  1000	                k: Field(data=v, name=f"d{k}_dt_phys",
  1001	                         dims=first.dT_dt.dims, units="kg/kg/s")
  1002	                for k, v in combined_tracer_tends.items()
  1003	            }
  1004	        combined_dv_dt = None
  1005	        if first.dv_dt is not None:
  1006	            combined_dv_dt = first.dv_dt.replace(data=dv_dt)
  1007	        return HydrostaticTendencies(
  1008	            du_dt=first.du_dt.replace(data=du_dt),
  1009	            dT_dt=first.dT_dt.replace(data=dT_dt),
  1010	            dp_s_dt=first.dp_s_dt.replace(data=dp_s_dt),
  1011	            dphis_dt=first.dphis_dt.replace(data=dphis_dt),
  1012	            dv_dt=combined_dv_dt,
  1013	            tracer_tendencies=tracer_tends_out,
  1014	        )
  1015	
  1016	    def _attach_sfc_precip(combined, first, precip_accum):
  1017	        """Carry surface precip [kg/m^2/s] on the combined tendency (dropped by
  1018	        _build_combined) so the lean MPAS loop can export it. No-op / byte-
  1019	        identical when microphysics produced no precip (precip_accum is None)."""
  1020	        if precip_accum is None:
  1021	            return combined
  1022	        return combined._replace(precip=Field(
  1023	            data=precip_accum, name="precip",
  1024	            dims=first.dp_s_dt.dims, units="kg/m^2/s"))
  1025	
  1026	    def _attach_sfc_diag_extras(combined, extras):
  1027	        """Carry the per-module surface/TOA diagnostic Fields (TOA trio from
  1028	        radiation, shflx/lhflx from turbulence) on the combined tendency —
  1029	        _build_combined constructs a fresh tendency that drops them. No-op /
  1030	        byte-identical when every extra is None (radiation+turbulence off)."""
  1031	        _set = {k: v for k, v in extras.items() if v is not None}
  1032	        return combined._replace(**_set) if _set else combined
  1033	
  1034	    # Diagnostic extras that are COUNTS (int): combined across macmic
  1035	    # sub-steps as the maximum, never weighted-averaged like the fluxes.
  1036	    _COUNT_DIAG_FIELDS = frozenset({"sed_substeps_required"})
  1037	    assert _COUNT_DIAG_FIELDS <= {"sw_up_toa", "lw_up_toa", "sw_down_toa",
  1038	                                  "shflx_sfc", "lhflx_sfc", "sw_up_toa_clr",
  1039	                                  "lw_up_toa_clr", "sed_substeps_required"}
  1040	
  1041	    def _advance_state(state, du_dt, dv_dt, dT_dt, dp_s_dt, tracer_tends, dt_x):
  1042	        """``state + dt_x * tendency`` (CAM ``physics_update``); ``phis`` fixed."""
  1043	        kw = dict(
  1044	            u=state.u.replace(data=state.u.data + dt_x * du_dt),
  1045	            T=state.T.replace(data=state.T.data + dt_x * dT_dt),
  1046	            p_s=state.p_s.replace(data=state.p_s.data + dt_x * dp_s_dt),
  1047	        )
  1048	        if state.v is not None and dv_dt is not None:
  1049	            kw["v"] = state.v.replace(data=state.v.data + dt_x * dv_dt)
  1050	        if state.tracers is not None and tracer_tends:
  1051	            kw["tracers"] = {
  1052	                k: (f.replace(data=f.data + dt_x * tracer_tends[k])
  1053	                    if k in tracer_tends else f)
  1054	                for k, f in state.tracers.items()
  1055	            }
  1056	        return state._replace(**kw)
  1057	
  1058	    def _add_scaled(acc, r, w):
  1059	        """``acc + w * r`` over the ``_accumulate`` result tuple (None-aware).
  1060	
  1061	        ``first`` keeps the first non-None; ``phys_updates`` merge (later
  1062	        wins, so a sub-cycled carry leaves the loop at its final value).
  1063	        """
  1064	        if acc is None:
  1065	            acc = (None, None, None, None, None, {}, {}, None, None,
  1066	                   {k: None for k in r[9]}, None)
  1067	        (du, dv, dT, dps, dphis, tr, upd, first, pr, ex, led) = acc
  1068	        (du_r, dv_r, dT_r, dps_r, dphis_r, tr_r, upd_r, first_r, pr_r,
  1069	         ex_r, led_r) = r
  1070	
  1071	        def _ws(a, b):
  1072	            if b is None:
  1073	                return a
  1074	            return w * b if a is None else a + w * b
  1075	
  1076	        tr = dict(tr)
  1077	        for k, v in tr_r.items():
  1078	            tr[k] = _ws(tr.get(k), v)
  1079	        upd = dict(upd)
  1080	        upd.update(upd_r)
  1081	        ex = dict(ex)
  1082	        for k, v in ex_r.items():
  1083	            if v is None:
  1084	                continue
  1085	            if k in _COUNT_DIAG_FIELDS:
  1086	                # a COUNT: the window's worst sub-step, not an average
  1087	                ex[k] = (v if ex.get(k) is None
  1088	                         else ex[k].replace(data=jnp.maximum(ex[k].data, v.data)))
  1089	                continue
  1090	            ex[k] = (v.replace(data=w * v.data) if ex.get(k) is None
  1091	                     else ex[k].replace(data=ex[k].data + w * v.data))
  1092	        return (_ws(du, du_r), _ws(dv, dv_r), _ws(dT, dT_r), _ws(dps, dps_r),
  1093	                _ws(dphis, dphis_r), tr, upd,
  1094	                first if first is not None else first_r,
  1095	                _ws(pr, pr_r), ex, _ws(led, led_r))
  1096	
  1097	    def _carry_within_call(ps, upd):
  1098	        """Intermediate carry update INSIDE one physics call: the per-call
  1099	        inputs (dynamics tendencies, prescribed surface fluxes) stay visible
  1100	        to every later sub-module; the call's final update clears them."""
  1101	        if ps is None:
  1102	            return None
  1103	        keep = {k: getattr(ps, k) for k in PHYSSTATE_PER_CALL_INPUTS}
  1104	        return update_physics_state(ps, {**keep, **upd})
  1105	
  1106	    def _accumulate_step(fns, groups, state, grid, sigma_coord, phys_state,
  1107	                         forcing, ledger_rows=None):
  1108	        """``_accumulate`` (N=1, byte-identical) or the CAM macmic loop."""
  1109	        if _n_macmic == 1:
  1110	            return _accumulate(fns, state, grid, sigma_coord, phys_state,
  1111	                               forcing, ledger_rows=ledger_rows)
  1112	        rows = ledger_rows if ledger_rows is not None else [None] * len(fns)
  1113	        sel = {g: [(f, r) for f, gg, r in zip(fns, groups, rows) if gg == g]
  1114	               for g in ("once", "pre", "sub")}
  1115	
  1116	        def _run(fns_rows, st, ps):
  1117	            return _accumulate([f for f, _ in fns_rows], st, grid, sigma_coord,
  1118	                               ps, forcing,
  1119	                               ledger_rows=([r for _, r in fns_rows]
  1120	                                            if ledger_rows is not None else None))
  1121	        # Evaluation order: the group holding ``fns[0]`` first, so ``first``
  1122	        # (radiation when configured) is the template exactly as at N=1.
  1123	        # ``first`` is only ever a Field TEMPLATE and, when radiation leads,
  1124	        # the radiative heating; on a sub-led list it is sub-step 1's dt/N
  1125	        # tendency and must not be read quantitatively.
  1400	        )
  1401	    if config.radiation.scheme != "none":
  1402	        # The spectral radiation builder has a READ side for the CLUBB
  1403	        # cloud-fraction carry (and the CAM6 deepcu carries); the flag
  1404	        # selects it, and the dispatcher below forwards ``phys_state`` to
  1405	        # the fn that advertises ``_wants_phys_state_ro``.
  1406	        tagged_fns.append((make_radiation_physics(
  1407	            config.radiation, "spectral_pe",
  1408	            sfc_albedo_override=sfc_albedo_override,
  1409	            sfc_emissivity_override=sfc_emissivity_override,
  1410	            use_clubb_cloud_fraction=config.radiation.use_clubb_cloud_fraction,
  1411	        ), False, None))
  1412	    if config.convection.scheme != "none":
  1413	        tagged_fns.append((make_convection_physics(config.convection, "spectral_pe", dt), True, "conv_prog_profile"))
  1414	    if config.turbulence.scheme != "none":
  1415	        # Phase C codex iter-2 high: route MYNN-2.5 to ``qke`` on the
  1416	        # spectral PE combined path too.
  1417	        # qke (MYNN-2.5) / clubb_moments (prognostic CLUBB) / tke — must match the
  1418	        # slot the per-model physics_fn reads (turbulence_carry_field).
  1419	        _turb_sn, _, _turb_sc = get_turbulence_fn(config.turbulence)
  1420	        _turb_field = turbulence_carry_field(_turb_sn, _turb_sc)
  1421	        tagged_fns.append((
  1422	            make_turbulence_physics(config.turbulence, "spectral_pe", dt),
  1423	            True,
  1424	            _turb_field,
  1425	        ))
  1426	    if config.microphysics.scheme != "none":
  1427	        tagged_fns.append((make_microphysics_physics(config.microphysics, "spectral_pe", dt), False, None))
  1428	    if config.gravity_wave_drag.scheme != "none":
  1429	        tagged_fns.append((make_gwd_physics(config.gravity_wave_drag, "spectral_pe", dt), True, "gwd_spectrum"))
  1430	
  1431	    def physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
  1432	        if not tagged_fns:
  1433	            zero_3d = jnp.zeros_like(state.vor_hat.data)
  1434	            zero_2d = jnp.zeros_like(state.lnps_hat.data)
  1435	            # Preserve the input state's tracer pytree structure as a
  1436	            # zero tendency so downstream tree.map(state, tendency)
  1437	            # works.  ``zero_like_tracers`` duck-types Field vs raw-array.
  1438	            zero_tracers = zero_like_tracers(state.tracers)
  1439	            zero_tend = SpectralHydrostaticState(
  1440	                vor_hat=state.vor_hat.replace(data=zero_3d),
  1441	                div_hat=state.div_hat.replace(data=zero_3d),
  1442	                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
  1443	                lnps_hat=state.lnps_hat.replace(data=zero_2d),
  1444	                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
  1445	                tracers=zero_tracers,
  1446	            )
  1447	            return zero_tend, None
  1448	
  1449	        # Compute grid-space diagnostics once and reuse across all active modules.
  1450	        shared_fields = spectral_pe_to_grid(state, grid, sigma_coord)
  1451	        phys_updates = {}
  1452	
  1453	        fn0, accepts_ps, field_name = tagged_fns[0]
  1454	        _fwd0 = ({"forcing": forcing}
  1455	                 if getattr(fn0, "_wants_forcing", False) else {})
  1456	        if accepts_ps:
  1457	            first, field_val = fn0(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state, **_fwd0)
  1458	            if field_val is not None and field_name is not None:
  1459	                # Multi-field updates (e.g., Bechtold's conv_prog_profile
  1460	                # and conv_stoch_state) are returned as a dict, which
  1461	                # we merge into ``phys_updates``.
  1462	                if isinstance(field_val, dict):
  1463	                    phys_updates.update(field_val)
  1464	                else:
  1465	                    phys_updates[field_name] = field_val
  1466	        else:
  1467	            if getattr(fn0, "_wants_phys_state_ro", False):
  1468	                # Read-only phys_state consumer (radiation reading the CLUBB
  1469	                # cloud-fraction / CAM6 deepcu carries): forward it but keep
  1470	                # the single-return contract -- no carry is written back.
  1471	                _fwd0 = {**_fwd0, "phys_state": phys_state}
  1472	            first = fn0(state, grid, sigma_coord, grid_fields=shared_fields, **_fwd0)
  1473	        vor_hat = first.vor_hat.data
  1474	        div_hat = first.div_hat.data
  1475	        T_hat = first.T_hat.data
  1476	        lnps_hat = first.lnps_hat.data
  1477	        phis_hat = first.phis_hat.data
  1478	
  1479	        # Tracer accumulation: start from the first module's
  1480	        # contribution if it has any, otherwise None.  Subsequent
  1481	        # modules are summed in the loop below.  Each value is kept
  1482	        # as a raw grid-space jax array (dropping the Field container)
  1483	        # so the per-key sum is a plain ``+``.  We re-wrap with the
  1484	        # state's container at the end for pytree-leaf consistency.
  1485	        def _grid_data(value):
  1486	            return value.data if hasattr(value, "data") else value
  1487	        accumulated_tracers = None
  1488	        if first.tracers is not None:
  1489	            accumulated_tracers = {
  1490	                k: _grid_data(v) for k, v in first.tracers.items()
  1491	            }
  1492	
  1493	        for fn, accepts_ps, field_name in tagged_fns[1:]:
  1494	            _fwd = ({"forcing": forcing}
  1495	                    if getattr(fn, "_wants_forcing", False) else {})
  1496	            if accepts_ps:
  1497	                t, field_val = fn(state, grid, sigma_coord, grid_fields=shared_fields, phys_state=phys_state, **_fwd)
  1498	                if field_val is not None and field_name is not None:
  1499	                    # Match the first-iteration branch: dict updates
  1500	                    # (e.g., Bechtold's conv_prog_profile +
  1501	                    # conv_stoch_state + prng_key) must MERGE into
  1502	                    # phys_updates, not be assigned as a single
  1503	                    # blob under field_name.
  1504	                    if isinstance(field_val, dict):
  1505	                        phys_updates.update(field_val)
  1506	                    else:
  1507	                        phys_updates[field_name] = field_val
  1508	            else:
  1509	                if getattr(fn, "_wants_phys_state_ro", False):
  1510	                    _fwd = {**_fwd, "phys_state": phys_state}
  1511	                t = fn(state, grid, sigma_coord, grid_fields=shared_fields, **_fwd)
  1512	            vor_hat = vor_hat + t.vor_hat.data
  1513	            div_hat = div_hat + t.div_hat.data
  1514	            T_hat = T_hat + t.T_hat.data
  1515	            lnps_hat = lnps_hat + t.lnps_hat.data
  1516	            phis_hat = phis_hat + t.phis_hat.data
  1517	            # Per-tracer accumulation across modules.
  1518	            if t.tracers is not None:
  1519	                if accumulated_tracers is None:
  1520	                    accumulated_tracers = {
  1521	                        k: _grid_data(v) for k, v in t.tracers.items()
  1522	                    }
  1523	                else:
  1524	                    for k, v in t.tracers.items():
  1525	                        v_data = _grid_data(v)
  1526	                        if k in accumulated_tracers:
  1527	                            accumulated_tracers[k] = (
  1528	                                accumulated_tracers[k] + v_data
  1529	                            )
  1530	                        else:
  1531	                            accumulated_tracers[k] = v_data
  1532	
  1533	        # Build the final tracers dict for the combined tendency,

codex
The cited gates and parallel accumulation are present. Piece A omits two parts of the MPAS contract: supplying existing cloud water to CLUBB and requiring both vapour and liquid tracers. I’m checking the ordering alternative and memory claim next.
exec
/usr/bin/bash -lc "nl -ba packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py | sed -n '1301,1485p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py | sed -n '6330,6368p;6475,6508p'; rg -n 'liquid_from_closure|2.47|2.5|def _make_spectral' packages/atmosphere/legoesm/atmosphere/physics/microphysics/{integration,morrison}.py; rg -n '643|nothing_saveable|cld_macmic_num_steps|clubb_liquid_partition|hard_sat_ice_curve' packages/*/legoesm/training scripts/train config/wb -g '*.py' -g '*.yaml' 2>/dev/null | head -65; rg --files | rg 'wb.*(build|rollout)|rollout.*py|spectral.*physics'" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 634ms:
  1301	def _make_spectral_pe_turbulence(
  1302	    turbulence_config: TurbulenceConfig,
  1303	    dt: float,
  1304	) -> Callable:
  1305	    """Create turbulence physics_fn for SpectralPEModel.
  1306	
  1307	    Signature: (state, grid, sigma_coord, grid_fields=None, phys_state=None)
  1308	               -> (SpectralHydrostaticState, tke_out)
  1309	
  1310	    When *phys_state* (a ``PhysicsState``) is passed, TKE is read from
  1311	    ``phys_state.tke`` and the updated TKE is returned as the second
  1312	    element of the result tuple.
  1313	    """
  1314	    scheme_name, turb_fn, scheme_config = get_turbulence_fn(turbulence_config)
  1315	    needs_tke = turbulence_scheme_traits(scheme_name).carries_energy
  1316	    carry_field = turbulence_carry_field(scheme_name, scheme_config)
  1317	    if carry_field == "qke":
  1318	        # MYNN-2.5 stays refused on this lane. The mechanism below is not
  1319	        # carry-specific, so it would probably work -- but "probably" is not a
  1320	        # measurement, and nothing exercises qke here. Lift it with its own
  1321	        # continuity test, not as a side effect of the CLUBB one.
  1322	        raise NotImplementedError(
  1323	            "MYNN-2.5 turbulence carries a prognostic qke on a spectral PE "
  1324	            "lane that has no test pinning the carry across steps. Use "
  1325	            "``model_type='hydrostatic'``, or add the continuity test "
  1326	            "(tests/unit/test_spectral_prognostic_clubb_carry.py is the "
  1327	            "pattern) before enabling it here.")
  1328	    # Prognostic CLUBB used to be refused alongside it, because the spectral
  1329	    # lane dropped the returned PhysicsState and the moments would have
  1330	    # re-initialised every step. What is true NOW, consumer by consumer:
  1331	    #   * the TRAINING rollout threads it: spectral_rollout scans
  1332	    #     (state, rad_cache, phys_state) whenever the physics_fn carries the
  1333	    #     with_phys_state / init_phys_state markers, which the classical
  1334	    #     factory attaches. This is the lane the AMIP-matched WB arm runs.
  1335	    #   * the MODEL's step() threads it on the LEAPFROG path only; under
  1336	    #     ssp_rk3 physics is evaluated per sub-stage, where a per-step carry is
  1337	    #     ill-defined, and ``refuse_unthreaded_stateful_physics`` still refuses
  1338	    #     there. That guard protects the RK3 model path, not this one.
  1339	    #   * a caller that threads NOTHING is refused per step below, rather than
  1340	    #     silently re-seeding.
  1341	    # The NONHYDROSTATIC lane keeps its own refusal: that driver still drops
  1342	    # the state.
  1343	
  1344	    def physics_fn(state, grid, sigma_coord, grid_fields=None, phys_state=None):
  1345	        tke_out = None
  1346	
  1347	        # Transform spectral state to grid space (or reuse precomputed fields).
  1348	        fields = grid_fields
  1349	        if fields is None:
  1350	            fields = spectral_pe_to_grid(state, grid, sigma_coord)
  1351	        u = fields['u']         # (n_lat, n_lon, nlev)
  1352	        v = fields['v']
  1353	        T = fields['T']
  1354	        p_s = fields['p_s']     # (n_lat, n_lon)
  1355	
  1356	        nlev = sigma_coord.n_levels
  1357	        n_lat, n_lon = p_s.shape
  1358	
  1359	        # Pressure at full and half levels
  1360	        p_full = sigma_coord.pressure_at_full(p_s)
  1361	        p_half = sigma_coord.pressure_at_half(p_s)
  1362	
  1363	        # Reshape to columns
  1364	        ncol = n_lat * n_lon
  1365	        T_col = T.reshape(ncol, nlev)
  1366	        u_col = u.reshape(ncol, nlev)
  1367	        v_col = v.reshape(ncol, nlev)
  1368	        p_full_col = p_full.reshape(ncol, nlev)
  1369	        p_half_col = p_half.reshape(ncol, nlev + 1)
  1370	        # Pin the column-physics dtype to the gridded state precision so
  1371	        # we never silently flow x64 zeros into the column path.
  1372	        _state_dtype = T.dtype
  1373	        # Water vapour from the state's tracer pytree, exactly as the
  1374	        # hydrostatic / MPAS / cubed-sphere bridges above do.  This lane
  1375	        # used to hand the scheme a hard-coded ZERO humidity column: the
  1376	        # turbulence closure then saw a bone-dry atmosphere (no moist
  1377	        # buoyancy in w'thv', a PDF cloud fraction computed from q_v = 0)
  1378	        # and its surface latent flux was evaluated against q_v = 0, i.e.
  1379	        # the largest moisture gradient the bulk formula can produce.
  1380	        # Clipped at zero on the way in, matching the microphysics bridge.
  1381	        if state.tracers is not None and "q_v" in state.tracers:
  1382	            _qv_raw = state.tracers["q_v"]
  1383	            _qv_data = _qv_raw.data if hasattr(_qv_raw, "data") else _qv_raw
  1384	            q_v_col = jnp.maximum(_qv_data.reshape(ncol, nlev), 0.0)
  1385	        else:
  1386	            q_v_col = jnp.zeros((ncol, nlev), dtype=_state_dtype)
  1387	
  1388	        zero_3d = jnp.zeros_like(state.vor_hat.data)
  1389	        zero_2d = jnp.zeros_like(state.lnps_hat.data)
  1390	
  1391	        if turb_fn is None:
  1392	            return SpectralHydrostaticState(
  1393	                vor_hat=state.vor_hat.replace(data=zero_3d),
  1394	                div_hat=state.div_hat.replace(data=zero_3d),
  1395	                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
  1396	                lnps_hat=state.lnps_hat.replace(data=zero_2d),
  1397	                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
  1398	            ), tke_out
  1399	
  1400	        # Heights and density (moist form — audit 2026-05-12 MEDIUM #8).
  1401	        z_full, z_half = _compute_heights_from_sigma(T_col, p_half_col, q_v=q_v_col)
  1402	        rho = _compute_rho(T_col, p_full_col, q_v=q_v_col)
  1403	
  1404	        T_sfc = _resolve_T_sfc(T_col, phys_state)
  1405	        q_sfc = saturation_mixing_ratio(T_sfc, p_full_col[:, -1])
  1406	        # A per-step prescribed surface flux (a diurnal cycle the run-constant
  1407	        # config scalar cannot carry) enters here; identity without one.
  1408	        step_config = _resolve_prescribed_surface_fluxes(
  1409	            scheme_config, phys_state, rho)
  1410	
  1411	        if needs_tke:
  1412	            if carry_field == "clubb_moments" and phys_state is None:
  1413	                # Seeding fresh here would look like a working run and be a
  1414	                # memoryless one: the moments would restart from the floor
  1415	                # every step. A caller on this lane must thread the carry (the
  1416	                # markers on the classical factory do).
  1417	                raise ValueError(
  1418	                    "prognostic CLUBB on the spectral PE lane was called "
  1419	                    "without a physics state, so its moments would re-seed "
  1420	                    "every step. Thread the carry (spectral_rollout does this "
  1421	                    "through the with_phys_state/init_phys_state markers), or "
  1422	                    "select diagnostic CLUBB.")
  1423	            tke_in = _read_turb_carry(
  1424	                phys_state, carry_field, ncol, nlev, scheme_config, _state_dtype)
  1425	            turb_out, tke_new = turb_fn(
  1426	                u_col, v_col, T_col, q_v_col, tke_in,
  1427	                p_full_col, p_half_col, z_full, z_half,
  1428	                T_sfc, q_sfc, rho, dt, step_config,
  1429	            )
  1430	            tke_out = tke_new
  1431	        else:
  1432	            # No land-flux hand-over on this lane (see the hydrostatic closure).
  1433	            turb_out = turb_fn(
  1434	                u_col, v_col, T_col, q_v_col,
  1435	                p_full_col, p_half_col, z_full, z_half,
  1436	                T_sfc, q_sfc, rho, dt, step_config,
  1437	            )
  1438	
  1439	        # Reshape tendencies to grid space
  1440	        du_dt = turb_out.du_dt.reshape(n_lat, n_lon, nlev)
  1441	        dv_dt = turb_out.dv_dt.reshape(n_lat, n_lon, nlev)
  1442	        dT_dt = turb_out.dT_dt.reshape(n_lat, n_lon, nlev)
  1443	
  1444	        # Project wind tendencies to spectral vorticity/divergence
  1445	        # (same pattern as held_suarez_forcing_spectral)
  1446	        a = grid.radius
  1447	        im_over_a = 1j * grid.ms.astype(jnp.float64) / a
  1448	        one_over_a = 1.0 / a
  1449	
  1450	        cos_lat_3d = grid.cos_lat[:, None, None]  # (n_lat, 1, 1)
  1451	        du_cos = du_dt * cos_lat_3d
  1452	        dv_cos = dv_dt * cos_lat_3d
  1453	
  1454	        # curl(du, dv) -> vorticity tendency
  1455	        dvor_hat = (
  1456	            im_over_a[:, None] * sh_analysis_oc2_3d(grid, dv_cos)
  1457	            + one_over_a * sh_analysis_dmu_3d(grid, du_cos)
  1458	        )
  1459	
  1460	        # div(du, dv) -> divergence tendency
  1461	        ddiv_hat = (
  1462	            im_over_a[:, None] * sh_analysis_oc2_3d(grid, du_cos)
  1463	            - one_over_a * sh_analysis_dmu_3d(grid, dv_cos)
  1464	        )
  1465	
  1466	        # Temperature tendency to spectral
  1467	        dT_hat = sh_analysis_3d(grid, dT_dt)
  1468	
  1469	        # Moisture tendency [kg/kg/s], positive = moistening, the same
  1470	        # sign convention the hydrostatic and MPAS bridges export.  Without
  1471	        # this the boundary layer moved momentum and heat but NOT water:
  1472	        # surface evaporation and vertical moisture mixing were computed by
  1473	        # the scheme and then discarded, so no turbulent flux of water
  1474	        # vapour reached the dycore on the spectral lane.  Untouched tracer
  1475	        # keys are mirrored as zeros so the combined wrapper's per-key
  1476	        # accumulation and the RK tree-map see a complete pytree.
  1477	        tracers_tend = None
  1478	        if state.tracers is not None:
  1479	            dq_v_dt = turb_out.dq_v_dt.reshape(n_lat, n_lon, nlev)
  1480	            tracers_tend = {}
  1481	            for name, template in state.tracers.items():
  1482	                if name == "q_v":
  1483	                    arr = dq_v_dt
  1484	                elif hasattr(template, "data"):
  1485	                    arr = jnp.zeros_like(template.data)
  6330	    if config.liquid_partition:
  6331	        if q_c is None:
  6332	            raise ValueError(
  6333	                "clubb_step: liquid_partition=True requires the host cloud "
  6334	                "liquid q_c. Without it the closure cannot seed rt = q_v + q_c, "
  6335	                "and replacing the host liquid with a closure that never saw it "
  6336	                "is a one-way condensation ratchet (clubb_intr.F90:1546,2160)."
  6337	            )
  6338	    elif q_c is not None:
  6339	        raise ValueError(
  6340	            "clubb_step: q_c was supplied but liquid_partition=False, so the "
  6341	            "cloud liquid would be silently ignored. Enable the partition or "
  6342	            "stop passing q_c."
  6343	        )
  6344	
  6345	    # ---- Thermodynamics (top-down) ----
  6346	    exner = exner_function(p_full)
  6347	    # CAM's two-sided liquid exchange (clubb_intr.F90:1546,1550): the conserved
  6348	    # variables the closure advances are TOTAL water and LIQUID-water potential
  6349	    # temperature, so the host's existing cloud liquid enters both. With the
  6350	    # lever off, q_c is absent and these reduce to the historical q_v / theta.
  6351	    if config.liquid_partition:
  6352	        q_t = q_v + q_c
  6353	        T_l = T - (constants.L_v / constants.c_pd) * q_c
  6354	        thv = virtual_potential_temperature_with_liquid(T, q_v, q_c, exner)
  6355	    else:
  6356	        q_t = q_v
  6357	        T_l = T
  6358	        thv = virtual_temperature(T, q_v) / exner
  6359	    theta = T_l / exner
  6360	
  6361	    # ---- Ascending CLUBB grid + means on zt ----
  6362	    gr = make_clubb_grid_from_levels(z_full, z_half)
  6363	    thlm = flip_vertical(theta)          # thl ~ theta (zt)
  6364	    rtm = flip_vertical(q_t)             # rt = q_v (+ q_c with the partition)
  6365	    um = flip_vertical(u)
  6366	    vm = flip_vertical(v)
  6367	    exner_zt = flip_vertical(exner)
  6368	    p_zt = flip_vertical(p_full)
  6475	            or getattr(config.surface, "prescribed_tau_y_pa", None) is not None))
  6476	
  6477	    # ---- Map advanced means back to top-down tendencies ----
  6478	    u_new = flip_vertical(new_state.um)
  6479	    v_new = flip_vertical(new_state.vm)
  6480	    du_dt = (u_new - u) / dt
  6481	    dv_dt = (v_new - v) / dt
  6482	    if config.liquid_partition:
  6483	        # CAM's outbound split (clubb_intr.F90:2159,2160): the post-advance PDF
  6484	        # closure's rcm IS the new cloud liquid, so vapour is the remainder of
  6485	        # total water and the temperature carries that liquid's latent heat.
  6486	        # rcm is clip_rcm'd against rtm inside the closure, so q_v stays >= 0.
  6487	        rcm_new = flip_vertical(diags["rcm_grid"])
  6488	        q_new = flip_vertical(new_state.rtm) - rcm_new
  6489	        T_new = (flip_vertical(new_state.thlm) * exner
  6490	                 + (constants.L_v / constants.c_pd) * rcm_new)
  6491	        # REPLACES the host liquid, matching the reference; the seeding above is
  6492	        # what makes that reversible rather than a one-way ratchet.
  6493	        diags = dict(diags, dq_c_dt=(rcm_new - q_c) / dt)
  6494	    else:
  6495	        T_new = flip_vertical(new_state.thlm) * exner  # thl ~ theta, no liquid
  6496	        q_new = flip_vertical(new_state.rtm)
  6497	    dT_dt = (T_new - T) / dt
  6498	    dq_v_dt = (q_new - q_v) / dt
  6499	    diags = dict(diags, ustar=ustar, shflx=shflx, lhflx=lhflx)
  6500	    return du_dt, dv_dt, dT_dt, dq_v_dt, new_state, diags
  6501	
  6502	
  6503	def clubb_turbulence_prognostic(
  6504	    u: jax.Array,
  6505	    v: jax.Array,
  6506	    T: jax.Array,
  6507	    q_v: jax.Array,
  6508	    clubb_moments: jax.Array,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:116:#   sigma-36 l30_trop_logstrat (bottom 281 m), dt = 112.5 s (production):
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:140:_NUC_T_THRESHOLD_K = 265.15      # heterogeneous ice-nucleation T threshold [K]
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:302:    if config.liquid_from_closure:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:390:    # f_ice transitions around ``cooper_T_act`` (≈ 265 K by default), NOT
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:404:    # air below 235 K may hold ice supersaturation up to rh_homo (gSAM
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:430:    # only where (RH_liq ≥ 0.999 AND T ≤ 265.15 K) OR RH_ice ≥ 1.08 — NOT
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:577:        # that wrongly suppressed warm-mixed-phase (265–273 K) ice growth
packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py:1081:def _make_spectral_pe_microphysics(
packages/ml/legoesm/training/neural_gcm_spectral.py:1623:                policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/neural_gcm_spectral.py:1638:        # ``prevent_cse=True`` plus ``policy=nothing_saveable`` is the
packages/ml/legoesm/training/neural_gcm_spectral.py:1647:            policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/neural_gcm_spectral.py:1935:                policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/neural_gcm_spectral.py:1941:            policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/neural_gcm_spectral.py:1962:        policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/neural_gcm_spectral.py:2128:            policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/neural_gcm_spectral.py:5107:        ``policy=nothing_saveable``) — the SAME gradient-checkpoint pattern the
packages/ml/legoesm/training/neural_gcm_spectral.py:5112:        SFNO. ``nothing_saveable`` recomputes every intermediate on the backward
packages/ml/legoesm/training/neural_gcm_spectral.py:5132:                policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/neural_gcm_spectral.py:5152:            policy=jax.checkpoint_policies.nothing_saveable,
packages/ml/legoesm/training/dycore_rollout.py
packages/ml/legoesm/ml/s2s/sfno_slab/rollout.py
tests/unit/test_multi_step_rollout_loss.py
tests/atmosphere/hydrostatic/unit/test_spectral_pe_microphysics.py
tests/unit/test_training_driver_rollout_hours.py
tests/unit/test_dycore_rollout_checkpoint.py
tests/unit/test_spectral_rollout_mass_anchor.py
tests/unit/test_wb_prescribed_flux_rollout.py
tests/unit/test_sfno_pe_rollout_stability.py

exec
/usr/bin/bash -lc "rg -n 'clubb_liquid_partition|cld_macmic_num_steps|hard_sat_ice_curve|hard_saturation_adjustment|make_physics\\(' packages/ml/legoesm/training/neural_gcm_spectral.py packages/core/legoesm/config*.py; nl -ba packages/atmosphere/legoesm/atmosphere/physics/microphysics/integration.py | sed -n '1110,1270p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py | sed -n '286,327p'; rg -n '643|arena' docs/wb --glob '"'!reviews/*'"' --glob '"'!**/*codex*'"' --glob '"'!**/*glm*'"' | head -40; rg -n '2.47|2.5|autoconv' packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison*.py | head -30; nl -ba config/wb/campaign/spectral_t63_cam6.yaml | sed -n '210,300p'" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 1603ms:
rg: packages/core/legoesm/config*.py: No such file or directory (os error 2)
packages/ml/legoesm/training/neural_gcm_spectral.py:948:    raw_fn = make_physics(cfg, model_type="spectral_pe", dt=dt)
packages/ml/legoesm/training/neural_gcm_spectral.py:1067:    raw_fn = make_physics(physics_config, model_type="spectral_pe", dt=dt,
  1110	        "q_i": "dq_i_dt",
  1111	        "q_s": "dq_s_dt",
  1112	        "q_g": "dq_g_dt",
  1113	        "N_c": "dN_c_dt",
  1114	        "N_r": "dN_r_dt",
  1115	        "N_i": "dN_i_dt",
  1116	    }
  1117	
  1118	    def physics_fn(state, grid, sigma_coord, grid_fields=None):
  1119	        # Transform spectral state to grid space
  1120	        fields = grid_fields
  1121	        if fields is None:
  1122	            fields = spectral_pe_to_grid(state, grid, sigma_coord)
  1123	        T = fields['T']
  1124	        p_s = fields['p_s']
  1125	
  1126	        nlev = sigma_coord.n_levels
  1127	        n_lat, n_lon = p_s.shape
  1128	
  1129	        zero_3d = jnp.zeros_like(state.vor_hat.data)
  1130	        zero_2d = jnp.zeros_like(state.lnps_hat.data)
  1131	        # Pin the column-physics dtype to the gridded state precision so
  1132	        # we do not silently flow x64 zeros into the column path.
  1133	        _state_dtype = T.dtype
  1134	
  1135	        if micro_fn is None:
  1136	            # Mirror the input tracer pytree shape with zeros so the
  1137	            # orchestrator's accumulator and the dycore RHS see a
  1138	            # consistent tendency structure even when microphysics is
  1139	            # disabled.
  1140	            return SpectralHydrostaticState(
  1141	                vor_hat=state.vor_hat.replace(data=zero_3d),
  1142	                div_hat=state.div_hat.replace(data=zero_3d),
  1143	                T_hat=state.T_hat.replace(data=jnp.zeros_like(state.T_hat.data)),
  1144	                lnps_hat=state.lnps_hat.replace(data=zero_2d),
  1145	                phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
  1146	                tracers=zero_like_tracers(state.tracers),
  1147	            )
  1148	
  1149	        # Pressure at full and half levels
  1150	        p_full = sigma_coord.pressure_at_full(p_s)
  1151	        p_half = sigma_coord.pressure_at_half(p_s)
  1152	
  1153	        # Reshape to columns
  1154	        ncol = n_lat * n_lon
  1155	        T_col = T.reshape(ncol, nlev)
  1156	        p_full_col = p_full.reshape(ncol, nlev)
  1157	        p_half_col = p_half.reshape(ncol, nlev + 1)
  1158	
  1159	        # Pull tracer fields out of ``state.tracers`` and reshape to the
  1160	        # column-physics ``(ncol, nlev)`` layout.  Backend microphysics
  1161	        # schemes assume non-negative mixing ratios, so clip on the way
  1162	        # in (matches the hydrostatic bridge's ``_get_tracer``).
  1163	        def _get_tracer(name):
  1164	            if state.tracers is not None and name in state.tracers:
  1165	                raw = state.tracers[name]
  1166	                data = raw.data if hasattr(raw, "data") else raw
  1167	                return jnp.maximum(data.reshape(ncol, nlev), 0.0)
  1168	            return jnp.zeros((ncol, nlev), dtype=_state_dtype)
  1169	
  1170	        q_v_col = _get_tracer("q_v")
  1171	
  1172	        # Moist (virtual-temperature) density/thickness — see the hydrostatic
  1173	        # bridge note above.
  1174	        rho = _compute_rho(T_col, p_full_col, q_v_col)
  1175	        dz = _compute_heights_from_sigma(T_col, p_half_col, q_v_col)
  1176	
  1177	        hydrometeors = HydrometeorState(
  1178	            q_c=_get_tracer("q_c"),
  1179	            q_r=_get_tracer("q_r"),
  1180	            q_i=_get_tracer("q_i"),
  1181	            q_s=_get_tracer("q_s"),
  1182	            q_g=_get_tracer("q_g"),
  1183	            # stored per MASS, used per VOLUME — see _PER_MASS_NUMBER_SPECIES
  1184	            N_c=number_per_mass_to_per_volume(_get_tracer("N_c"), rho),
  1185	            N_r=number_per_mass_to_per_volume(_get_tracer("N_r"), rho),
  1186	            N_i=_get_tracer("N_i"),
  1187	        )
  1188	
  1189	        if is_ml:
  1190	            if _ml_model_cache[0] is None:
  1191	                key = jax.random.PRNGKey(scheme_config.seed)
  1192	                _ml_model_cache[0] = MicrophysicsEmulator(
  1193	                    scheme_config.n_input, scheme_config.n_hidden,
  1194	                    scheme_config.n_layers, scheme_config.n_output, key=key,
  1195	                )
  1196	            micro_out = micro_fn(
  1197	                T_col, q_v_col, hydrometeors,
  1198	                p_full_col, p_half_col, rho, dz, dt,
  1199	                scheme_config, _ml_model_cache[0],
  1200	            )
  1201	        else:
  1202	            micro_out = micro_fn(
  1203	                T_col, q_v_col, hydrometeors,
  1204	                p_full_col, p_half_col, rho, dz, dt, scheme_config,
  1205	            )
  1206	
  1207	        dT_dt = micro_out.dT_dt.reshape(n_lat, n_lon, nlev)
  1208	
  1209	        # Transform T tendency to spectral space
  1210	        dT_hat = sh_analysis_3d(grid, dT_dt)
  1211	
  1212	        # Build the tracer tendency dict in grid-space ``(n_lat, n_lon,
  1213	        # nlev)`` layout, matching ``SpectralHydrostaticState.tracers``.
  1214	        # Wrap each tendency back into the same container type as the
  1215	        # input state's tracer (``Field`` vs raw ``jax.Array``) so the
  1216	        # SSP-RK ``tree.map`` pytree leaves line up.  Untouched tracer
  1217	        # keys are mirrored as zeros via ``zero_like_tracers``.
  1218	        tracers_tend = None
  1219	        if state.tracers is not None:
  1220	            tt = {}
  1221	            for name, attr in _TRACER_TEND_MAP.items():
  1222	                if name not in state.tracers:
  1223	                    continue
  1224	                template = state.tracers[name]
  1225	                _tend_col = getattr(micro_out, attr)
  1226	                if name in _PER_MASS_NUMBER_SPECIES:
  1227	                    # Back to the per-MASS storage the dycore transports.
  1228	                    _tend_col = number_per_volume_to_per_mass(_tend_col, rho)
  1229	                tend_grid = _tend_col.reshape(
  1230	                    n_lat, n_lon, nlev,
  1231	                )
  1232	                if hasattr(template, "data") and hasattr(template, "replace"):
  1233	                    tt[name] = template.replace(
  1234	                        data=tend_grid.astype(template.data.dtype),
  1235	                    )
  1236	                else:
  1237	                    tt[name] = tend_grid.astype(template.dtype)
  1238	            # Mirror any untouched tracer keys (e.g. a passive scalar
  1239	            # the user attached) as zeros so the orchestrator's
  1240	            # accumulator and the dycore RHS see a complete pytree.
  1241	            zeros = zero_like_tracers(state.tracers)
  1242	            if zeros is not None:
  1243	                for k, zv in zeros.items():
  1244	                    tt.setdefault(k, zv)
  1245	            tracers_tend = tt
  1246	
  1247	        return SpectralHydrostaticState(
  1248	            vor_hat=state.vor_hat.replace(data=zero_3d),
  1249	            div_hat=state.div_hat.replace(data=zero_3d),
  1250	            T_hat=state.T_hat.replace(data=dT_hat),
  1251	            lnps_hat=state.lnps_hat.replace(data=zero_2d),
  1252	            phis_hat=state.phis_hat.replace(data=jnp.zeros_like(state.phis_hat.data)),
  1253	            tracers=tracers_tend,
  1254	        )
  1255	
  1256	    def reset_state():
  1257	        _ml_model_cache[0] = None
  1258	
  1259	    physics_fn.reset_state = reset_state
  1260	    return physics_fn
   286	    # vapour-to-liquid CONDENSATION (micro_mg_cam.F90:668-672 switches the
   287	    # residual block at micro_mg2_0.F90:2688-2730 off under CLUBB).
   288	    #
   289	    # POSITIVE BRANCH ONLY.  An earlier version of this gate zeroed the signed
   290	    # rate, which was wrong twice over.  CAM's residual block is guarded by
   291	    # ``qtmp > qvn`` at micro_mg2_0.F90:2700, i.e. it fires only on positive
   292	    # supersaturation and has no evaporation branch to switch off in the first
   293	    # place.  And our default ``wbf_scheme="emergent"`` has NO explicit
   294	    # Bergeron rate (see the WBF section below): the mixed-phase cloud-water
   295	    # sink IS the negative branch here, evaporating liquid as ice deposition
   296	    # draws vapour below liquid saturation.  Zeroing it deleted that sink
   297	    # outright on any ice-supersaturated, liquid-subsaturated cell.
   298	    #
   299	    # ``q_sat`` is untouched -- rain evaporation, the sub-grid cloud-fraction
   300	    # closure and the ice branch all still read it.  Static Python gate on a
   301	    # config bool, the documented feature-gating exception to ``jnp.where``.
   302	    if config.liquid_from_closure:
   303	        condensation = jnp.minimum(condensation, 0.0)
   304	    # Sub-grid in-cloud closure (Morrison & Gettelman 2008): evaluate the
   305	    # warm-rain rates on the IN-CLOUD water q_c/cf and scale back by cf, so the
   306	    # non-linear KK2000/SB rates see the (higher) in-cloud concentration rather
   307	    # than the grid-mean.  cf is the Sundqvist √-form from the local RH, floored
   308	    # for AD/numeric safety.  When disabled, cf_eff=1 ⇒ identity (grid-mean).
   309	    if getattr(config, "subgrid_autoconversion", False):
   310	        RH = q_v / jnp.maximum(q_sat, 1.0e-10)
   311	        arg = (1.0 - RH) / max(1.0 - config.subgrid_rh_crit, 1.0e-6)
   312	        arg_safe = jnp.where(arg > 0.0, arg, 1.0)
   313	        cf_sg = jnp.where(arg > 0.0, 1.0 - jnp.sqrt(arg_safe), 1.0)
   314	        cf_eff = jnp.clip(cf_sg, config.subgrid_cf_min, 1.0)
   315	    else:
   316	        cf_eff = jnp.ones_like(q_c)
   317	    q_c_ic = q_c / cf_eff
   318	    q_r_ic = q_r / cf_eff
   319	    # Warm-rain autoconversion + accretion. KK2000 (default) is the SAM
   320	    # M2005 oracle scheme; Seifert-Beheng retained for back-compat.
   321	    if config.warm_rain_scheme == "kk2000":
   322	        dq_c_au, dN_r_au, x_c = autoconversion_kk2000(q_c_ic, N_c_eff, rho, dt)
   323	        dq_c_au = dq_c_au * cf_eff
   324	        dN_r_au = dN_r_au * cf_eff
   325	        dq_c_ac = accretion_kk2000(q_c_ic, q_r_ic) * cf_eff
   326	    elif config.warm_rain_scheme == "seifert_beheng":
   327	        dq_c_au, dN_r_au, x_c = autoconversion_sb(
docs/wb/reviews/2026-09-24_clubb_liquid_spectral_claim.md:34:    sub-step, and the WB gradient's memory arena is ALREADY 643 GiB on one GPU before this), or
18:- NPRCI (ice→snow autoconversion number, SAM :3323-3326): verified WIRED
19:  as ``dN_i_autoconv`` (= PRCI/CONS22 capped at N_i/dt) — an N_i sink and,
63:    autoconversion_sb,
64:    autoconversion_sb2001,
65:    autoconversion_kk2000,
116:#   sigma-36 l30_trop_logstrat (bottom 281 m), dt = 112.5 s (production):
140:_NUC_T_THRESHOLD_K = 265.15      # heterogeneous ice-nucleation T threshold [K]
159:    ``ice_to_snow_scheme`` m2005_autoconv mg_ferrier (180-s Ferrier prci)
309:    if getattr(config, "subgrid_autoconversion", False):
319:    # Warm-rain autoconversion + accretion. KK2000 (default) is the SAM
322:        dq_c_au, dN_r_au, x_c = autoconversion_kk2000(q_c_ic, N_c_eff, rho, dt)
327:        dq_c_au, dN_r_au, x_c = autoconversion_sb(
329:            config.autoconversion_sharpness,
341:        # q_c^4 autoconversion gets a cf^-3 enhancement vs cf^-1 for accretion —
344:        # ``subgrid_autoconversion=False`` (cf_eff=1). The SB2001-specific
345:        # cloud-NUMBER autoconv factor (2·PRC·rho/x_*) is not applied — the
347:        # instead; see ``autoconversion_sb2001``.
348:        dq_c_au, dN_r_au, x_c = autoconversion_sb2001(
390:    # f_ice transitions around ``cooper_T_act`` (≈ 265 K by default), NOT
404:    # air below 235 K may hold ice supersaturation up to rh_homo (gSAM
430:    # only where (RH_liq ≥ 0.999 AND T ≤ 265.15 K) OR RH_ice ≥ 1.08 — NOT
520:    # SAM PRCI ice→snow autoconversion (set inside the m2005 deposition block,
526:    # Ice-PSD slope/intercept (LAMI/N0I) and the snow-autoconversion size
528:    # but the ice->snow autoconversion (m2005_autoconv / mg_ferrier) needs
532:    # mg_ferrier autoconversion (the morrison_flavor="mg" default pairs
577:        # that wrongly suppressed warm-mixed-phase (265–273 K) ice growth
612:        # PRCI: ice→snow autoconversion — depositional growth of the cloud-ice
621:        # it is ``dN_i_autoconv`` below — an N_i sink AND (double-moment
631:        # used only inside LAMI, so gate the autoconv on ACTUAL ice (codex
707:    # allocation when PSACWS is much faster than autoconv/accretion). Warm-
   210	optimizer: adamw
   211	lr: 0.0015
   212	warmup_steps: 200
   213	weight_decay: 1.0e-05
   214	grad_clip_norm: 1.0
   215	grad_accum: 1
   216	loss:
   217	  w_T: 1.0
   218	  w_u: 0.5
   219	  w_v: 0.5
   220	  w_q: 0.3
   221	  w_ps: 0.3
   222	  w_bias_T: 100.0
   223	  w_bias_u: 1.0
   224	  w_bias_v: 1.0
   225	  w_bias_ps: 1.0
   226	  w_crps_T: 0.5
   227	  w_crps_u: 0.25
   228	  w_crps_v: 0.25
   229	  w_crps_q: 0.5
   230	  w_crps_ps: 0.15
   231	  w_spec_crps_T: 0.0
   232	  w_spec_crps_u: 0.0
   233	  w_spec_crps_v: 0.0
   234	  w_spec_crps_q: 0.0
   235	  multi_step_hours:
   236	  - 6
   237	  - 12
   238	  multi_step_weights:
   239	  - 1.0
   240	  - 1.0
   241	  level_weighting: pressure
   242	  normalize_by_scale: true
   243	  T_scale: 30.0
   244	  wind_scale: 20.0
   245	  q_scale: 0.005
   246	  ps_scale: 1000.0
   247	neural_gcm:
   248	  spatial_embedding: false
   249	  surface_drag_confounded: builder_refuses_classical_scheme
   250	  nn_hidden: 256
   251	  nn_layers: 4
   252	sfno:
   253	  spatial_embedding: false
   254	  sfno_embed_dim: 256
   255	  sfno_n_blocks: 8
   256	classical:
   257	  convection: zhang_mcfarlane
   258	  turbulence: clubb
   259	  cloud: cam6_clubb
   260	  microphysics: morrison
   261	  gwd: mcfarlane
   262	  surface_bulk: constant
   263	  rad_update_interval_steps: 6
   264	  orbital_insolation: true
   265	  convective_rain_to_surface: true
   266	  rrtmgp_gpoint_batch_size: 8
   267	  param_init:
   268	    atm.clouds.CloudConfig.rh_crit: 0.85
   269	    atm.clouds.CloudConfig.q_c_diagnostic: 5.0e-06
   270	  param_fixed:
   271	    atm.clouds.CloudConfig:
   272	      cloud_optics_inhomogeneity: two_region
   273	      cloud_partial_coverage_optics: none
   274	      cloud_vertical_overlap_optics: max_random
   275	      cloud_n_subcolumns: 8
   276	      cloud_fsd: 1.0
   277	      convective_cloud: false
   278	      saturation_scheme: mixed_phase
   279	    atm.turb.CLUBBConfig:
   280	      prognostic: true
   281	    atm.micro.MorrisonConfig:
   282	      # CAM6 clubb_do_liqsupersat = .false.: no condensation step after the
   283	      # physics, and no ice-curve blend of it. Matches the MPAS production
   284	      # deck (config/amip/amip_production.yaml, both rows false).
   285	      hard_saturation_adjustment: false
   286	      homogeneous_ice_nucleation: true
   287	      sed_cfl_substeps: true
   288	      sed_cfl_substeps_max: 256
   289	      sed_cfl_substeps_strict: false
   290	  clubb_top_press_hpa: 150.0

exec
/usr/bin/bash -lc "rg -n 'hard_saturation_adjustment|hard_sat_ice_curve|liquid_from_closure|clubb_liquid_partition' packages --glob '*.py' | head -95; rg -n '643|690[0-9]{6}|memory_analysis' scripts docs/wb --glob '"'!**/reviews/**'"' | head -35; nl -ba packages/ml/legoesm/training/neural_gcm_spectral.py | sed -n '1045,1080p;1920,1968p'; rg -n 'def autoconversion_kk2000|2.47' packages/atmosphere; rg -n 'param_fixed|make_classical|morrison_flavor' packages/ml/legoesm/training/wb* scripts/train* --glob '*.py' | head -35" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 2815ms:
packages/coupler/legoesm/driver/physics_pipeline.py:4094:    if getattr(config, "hard_saturation_adjustment", False):
packages/coupler/legoesm/driver/physics_pipeline.py:4095:        if "hard_saturation_adjustment" not in getattr(
packages/coupler/legoesm/driver/physics_pipeline.py:4098:                f"hard_saturation_adjustment=True is not supported by the "
packages/coupler/legoesm/driver/physics_pipeline.py:4103:        micro_config = micro_config._replace(hard_saturation_adjustment=True)
packages/coupler/legoesm/driver/physics_pipeline.py:4124:    # liquid_from_closure=True from reaching a built model with no liquid
packages/coupler/legoesm/driver/physics_pipeline.py:4136:        liquid_from_closure=_liq_closure,
packages/coupler/legoesm/driver/physics_pipeline.py:4392:        if getattr(config, "clubb_liquid_partition", False):
packages/coupler/legoesm/driver/physics_pipeline.py:4395:                    f"clubb_liquid_partition=True requires turbulence='clubb', "
packages/coupler/legoesm/driver/physics_pipeline.py:4400:                    "clubb_liquid_partition=True requires clubb_prognostic=True: "
packages/coupler/legoesm/driver/physics_pipeline.py:4503:    if getattr(config, "clubb_liquid_partition", False):
packages/coupler/legoesm/driver/physics_pipeline.py:4511:                "clubb_liquid_partition=True but an explicit turbulence_override "
packages/coupler/legoesm/driver/config.py:301:    # guard, same pattern as hard_sat_ice_curve).  Appended at the tuple END
packages/coupler/legoesm/driver/config.py:686:    clubb_liquid_partition: bool = False
packages/coupler/legoesm/driver/config.py:876:    hard_saturation_adjustment: bool = False
packages/coupler/legoesm/driver/config.py:879:    # without hard_saturation_adjustment=True is refused (silently-inert
packages/coupler/legoesm/driver/config.py:888:    # first ClimateEval scorecard.  Requires hard_saturation_adjustment.
packages/coupler/legoesm/driver/config.py:889:    hard_sat_ice_curve: bool = False
packages/coupler/legoesm/driver/config.py:1832:        if self.clubb_liquid_partition:
packages/coupler/legoesm/driver/config.py:2856:                    "clubb_liquid_partition=True needs cld_macmic_num_steps>=2: "
packages/coupler/legoesm/driver/config.py:2898:                        "clubb_liquid_partition=True leaves a diagnostic "
packages/coupler/legoesm/driver/config.py:3107:        if self.hard_sat_ice_curve:
packages/coupler/legoesm/driver/config.py:3108:            if not self.hard_saturation_adjustment:
packages/coupler/legoesm/driver/config.py:3110:                    "hard_sat_ice_curve=True requires "
packages/coupler/legoesm/driver/config.py:3111:                    "hard_saturation_adjustment=True (the mixed-phase curve "
packages/coupler/legoesm/driver/config.py:3117:                    "hard_sat_ice_curve=True requires an MPAS grid (the "
packages/coupler/legoesm/driver/config.py:3124:                    "hard_sat_ice_curve=True requires microphysics='morrison' "
packages/coupler/legoesm/driver/config.py:3134:        # hard_sat_ice_curve above and hard_saturation_adjustment below).
packages/coupler/legoesm/driver/config.py:3154:            if not self.hard_saturation_adjustment:
packages/coupler/legoesm/driver/config.py:3157:                    "hard_saturation_adjustment=True (the override would be "
packages/coupler/legoesm/driver/config.py:3181:        if (self.hard_saturation_adjustment
packages/coupler/legoesm/driver/config.py:3185:                "hard_saturation_adjustment requires a microphysics scheme "
packages/coupler/legoesm/driver/model_driver.py:9831:                    liquid_from_closure=cfg._liquid_partition_resolved(),
packages/coupler/legoesm/driver/model_driver.py:9832:                    # NOTE: hard_saturation_adjustment is INTENTIONALLY NOT
packages/coupler/legoesm/driver/model_driver.py:9870:        _hard_sat_on = bool(getattr(cfg, "hard_saturation_adjustment", False))
packages/coupler/legoesm/driver/model_driver.py:9874:            and "hard_saturation_adjustment" in getattr(_hsub, "_fields", ()))
packages/coupler/legoesm/driver/model_driver.py:9877:                f"hard_saturation_adjustment=True is not supported by the "
packages/coupler/legoesm/driver/model_driver.py:9895:        # hard_saturation_adjustment + grid_type='mpas' + microphysics='morrison'
packages/coupler/legoesm/driver/model_driver.py:9897:        _hard_sat_ice_curve = bool(
packages/coupler/legoesm/driver/model_driver.py:9898:            getattr(cfg, "hard_sat_ice_curve", False)) and _hard_sat_on
packages/coupler/legoesm/driver/model_driver.py:9907:        if _hard_sat_ice_curve and _hsub is not None:
packages/coupler/legoesm/driver/model_driver.py:11949:                _qi_fld = _trc.get("q_i") if _hard_sat_ice_curve else None
packages/coupler/legoesm/driver/model_driver.py:11955:                    ice_curve=_hard_sat_ice_curve,
packages/coupler/legoesm/driver/model_driver.py:11964:                            ", ice curve" if _hard_sat_ice_curve else "",
packages/atmosphere/legoesm/atmosphere/physics/combined.py:119:def _micro_liquid_from_closure_on(microphysics_config) -> bool:
packages/atmosphere/legoesm/atmosphere/physics/combined.py:132:    return bool(getattr(sub, "liquid_from_closure", False))
packages/atmosphere/legoesm/atmosphere/physics/combined.py:135:def _clubb_liquid_partition_on(turbulence_config) -> bool:
packages/atmosphere/legoesm/atmosphere/physics/combined.py:272:    if (_micro_liquid_from_closure_on(config.microphysics)
packages/atmosphere/legoesm/atmosphere/physics/combined.py:273:            and not _clubb_liquid_partition_on(config.turbulence)):
packages/atmosphere/legoesm/atmosphere/physics/combined.py:275:            "MorrisonConfig.liquid_from_closure=True without the CLUBB liquid "
packages/atmosphere/legoesm/atmosphere/physics/combined.py:279:            "(ExperimentConfig.clubb_liquid_partition drives them), or "
packages/atmosphere/legoesm/atmosphere/physics/combined.py:281:    if _clubb_liquid_partition_on(config.turbulence) and cld_macmic_num_steps < 2:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/p3.py:220:        hard_adjust=config.hard_saturation_adjustment,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/seifert_beheng.py:160:        hard_adjust=config.hard_saturation_adjustment,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:280:        hard_adjust=config.hard_saturation_adjustment,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:302:    if config.liquid_from_closure:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py:77:        "(config.hard_saturation_adjustment, default False) constrains ONLY "
packages/atmosphere/legoesm/atmosphere/physics/microphysics/ml_emulator.py:206:    if config.hard_saturation_adjustment:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/kessler.py:172:        hard_adjust=config.hard_saturation_adjustment,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/sundqvist.py:38:  * OPT-IN ONLY (``SundqvistConfig.hard_saturation_adjustment``, default
packages/atmosphere/legoesm/atmosphere/physics/microphysics/sundqvist.py:157:    if config.hard_saturation_adjustment:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/thompson.py:195:        hard_adjust=config.hard_saturation_adjustment,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:46:#   hard_saturation_adjustment : bool  -- opt-in gate; False => byte-identical
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:50:    "hard_saturation_adjustment",
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:55:    "hard_saturation_adjustment": False,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:427:    hard_saturation_adjustment: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:491:    hard_saturation_adjustment: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:520:    hard_saturation_adjustment: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:890:    hard_saturation_adjustment: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:926:    liquid_from_closure: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1069:    hard_saturation_adjustment: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1160:    hard_saturation_adjustment: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1201:    hard_saturation_adjustment: bool = False
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1252:    hard_saturation_adjustment: bool = False,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1259:    liquid_from_closure: bool | None = None,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1282:    nc_from_aerosol, subgrid_autoconversion, hard_saturation_adjustment : bool
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1316:    if hard_saturation_adjustment:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1317:        if "hard_saturation_adjustment" not in fields:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1319:                f"hard_saturation_adjustment=True is not supported by the "
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1327:        scheme_config = scheme_config._replace(hard_saturation_adjustment=True)
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1361:    if liquid_from_closure is not None and not hasattr(
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1362:            scheme_config, "liquid_from_closure"):
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1366:        if liquid_from_closure:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1368:                f"liquid_from_closure=True is not supported by the {scheme!r} "
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1371:    elif liquid_from_closure is not None:
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:1383:            liquid_from_closure=bool(liquid_from_closure))
scripts/plot/plot_scaling_paper_figure.py:159:             "L47\n(np\u226432\u2013128 packed on one node)",
scripts/validate/ec_site_regression_golden.json:82:      "beta_eff": 0.6904938120049127
scripts/validate/ec_site_regression_golden.json:163:      "beta_eff": 0.6764811676431104
scripts/validate/ec_site_regression_golden.json:208:      "beta_eff": 0.46893156901589766
scripts/validate/ec_site_regression_golden.json:487:      "beta_eff": 0.6603079923114643
scripts/validate/ec_site_regression_golden.json:594:      "gpp": 0.0001216438701038004,
scripts/validate/ec_site_regression_golden.json:802:      "beta_eff": 0.42035764366351946
scripts/bench/census_xla_dump.py:17:    process local) devices is not allowed. Got jax.Array: int32[2,2,1643]
scripts/validate/fv3_native/gen_duogrid_oracle.sh:5:# luanfs/FV3_container @ 7d06431e, see duogrid_ref/SOURCE_COMMIT.txt)
scripts/run/global_overturning/run_global_overturning_realistic_100yr_continuation.py:6:will be ``restart_day016438.npz`` — yr 45.04 — because the original
scripts/run/global_overturning/run_global_overturning_realistic_100yr_continuation.py:291:    # 16438 restart is at yr 45.04 (close enough to 45 — keep it as is).
scripts/validate/fv3_native/gen_dsw6_duo_oracle.py:77:        "3e102b7433b88664d8b31d159251e26aa2a4651085a1ab1c606643c1913ab7cf;"
scripts/validate/fv3_native/gen_dsw3_duo_oracle.py:68:        "3e102b7433b88664d8b31d159251e26aa2a4651085a1ab1c606643c1913ab7cf;"
scripts/validate/fv3_native/duogrid_oracle_driver.F90:4:! mirror luanfs/FV3_container @ 7d06431e — the module legoESM's duogrid.py /
scripts/validate/fv3_native/duogrid_oracle_driver_n2.F90:4:! mirror luanfs/FV3_container @ 7d06431e — the module legoESM's duogrid.py /
scripts/validate/fv3_native/duogrid_shim.F90:2:! Mouallem duo-grid variant, luanfs/FV3_container @ 7d06431e) compiles
scripts/validate/fv3_native/gen_dsw5_duo_oracle.py:96:        "3e102b7433b88664d8b31d159251e26aa2a4651085a1ab1c606643c1913ab7cf;"
scripts/validate/fv3_native/gen_geopk_pgrad_oracle.py:166:    "5e21f4b1ef8a5640aff643193d9c1ae36a48508dd17aef172f840ead11b9e353",
scripts/validate/ocean_fidelity/dino_1226/PREREG_tcarry_basin_reverdict.md:234:`35332f72823b76127972d76b039e039fd5c8de36db71ae42b64349cc89f7b1a1`.
scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py:1045:                                                              "row; before 0.997795757/1.006438108] "
scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py:1551:                                                              "measured corr=0.999643/ratio=0.999101 on synthetic "
scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round24.py:27:ROUND23_SHA = "4e9c6ee0a172ff5f5005186be1f1f7080927643ad88e1e7d044f00167c168e2e"
scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round32.py:31:ROUND31_SHA = "7ce94a68528e0e1b0fe950eeaa1ca194bcd9bbdfa0c8545459bbd7141888643d"
scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round38.py:35:CAPTURE_SHA = "96e11403c547c2bfa59f3871de7a099db82a66127d952006433ab34a55f36048"
scripts/validate/ocean_fidelity/dino_1226/PREREG_endwall_wind_placement.md:374:| first-eight ratio | `0.0979739110297185` / `0.10324729343288278` | `[1.0470043852687352,1.2805669019825299]` |
scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round55.py:107:    "stp_dump_23_after_traldf_tem.bin": "ff109b54b8f04c9dfcb09643b940b29f603fceff48d1896a510ba9af74a96f2f",
scripts/validate/ocean_fidelity/dino_1226/zdf_chain_end.py:55:ROW28_DUMP_SHA256 = "d3a62643bc8e5ea0370a784a6659cc386410baa516b270f86cacedd0603edad9"
scripts/validate/ocean_fidelity/dino_1226/probe_bottom_drag.py:13:correction to the barotropic forcing (``dynspg_ts.F90:1614-1643``, DINO
scripts/validate/ocean_fidelity/dino_1226/stpmlf_call_coverage.py:221:              "d3a62643bc8e5ea0370a784a6659cc386410baa516b270f86cacedd0603edad9 "
scripts/validate/ocean_fidelity/dino_1226/PHASE1_valid_stack_southern_budget.md:134:| 30-40 | +2.110 | +1.643 | +0.467 |
scripts/validate/ocean_fidelity/dino_1226/zdf_wall_epoch_group_score.py:29:    "ratio_first8": (1.0470043852687352, 1.2805669019825299),
  1045	    else:
  1046	        raise ValueError(
  1047	            f"unsupported radiation {radiation!r} for the trainable-physics "
  1048	            f"spectral path: must be one of ('rrtmgp', 'gray')")
  1049	
  1050	    # Build SBM convection config with trainable timescale + RH
  1051	    sbm_cfg = SBMConfig(
  1052	        tau_c=p.get('sbm_tau_c', 7200.0),
  1053	        rh_ref=p.get('sbm_RH_ref', 0.7),
  1054	    )
  1055	    conv_cfg = ConvectionConfig(scheme="sbm", sbm=sbm_cfg)
  1056	
  1057	    physics_config = PhysicsConfig(radiation=rad_cfg, convection=conv_cfg)
  1058	    # The trainable SURFACE leaves reach rrtmgp as per-call overrides — writing
  1059	    # them into RRTMGPConfig would key its solver cache on a tracer. Without
  1060	    # this they are inert on the rrtmgp path (codex): gray consumed the blended
  1061	    # albedo directly, so flipping the backend silently froze albedo_ocean /
  1062	    # albedo_ice. C_H / C_E ride the surface-flux config, not radiation.
  1063	    _albedo = p.get("albedo_ocean")
  1064	    _kwargs = {}
  1065	    if _albedo is not None and radiation == "rrtmgp":
  1066	        _kwargs["sfc_albedo_override"] = _albedo
  1067	    raw_fn = make_physics(physics_config, model_type="spectral_pe", dt=dt,
  1068	                          **_kwargs)
  1069	
  1070	    def physics_fn(state, grid_, sigma_coord):
  1071	        result = raw_fn(state, grid_, sigma_coord)
  1072	        return result[0] if isinstance(result, tuple) else result
  1073	
  1074	    return physics_fn
  1075	
  1076	
  1077	# =============================================================================
  1078	# Differentiable spectral rollout
  1079	# =============================================================================
  1080	
  1920	        # zero seed too; this is a deliberate departure for the training lane,
  1921	        # where the transient would be half of every sample.
  1922	        # The WHOLE carry from that pass is kept (owner decision 2026-09-24,
  1923	        # after the reviewers split): the alternative -- taking only the
  1924	        # three fields radiation reads -- would hand step 0 a carry whose
  1925	        # cloud fraction was diagnosed from moments the carry then does not
  1926	        # hold.  The cost of the whole pass is one extra relaxation step of
  1927	        # every prognostic carry (CLUBB moments, convection profile, GWD
  1928	        # spectrum) and one extra draw of the stochastic key, on an
  1929	        # atmosphere that has not moved.
  1930	        if phys_state_in is None and getattr(
  1931	                rad_physics_fn, "_wants_phys_state_ro", False):
  1932	            _warm = jax.checkpoint(
  1933	                lambda ps: _ps_entry(initial_state, grid, sigma_coord, ps)[1],
  1934	                prevent_cse=True,
  1935	                policy=jax.checkpoint_policies.nothing_saveable,
  1936	            )
  1937	            phys0 = _warm(phys0)
  1938	        step_fn_ckpt = jax.checkpoint(
  1939	            step_fn_gated_stateful,
  1940	            prevent_cse=True,
  1941	            policy=jax.checkpoint_policies.nothing_saveable,
  1942	        )
  1943	        (final_state, _, final_ps), _ = jax.lax.scan(
  1944	            step_fn_ckpt,
  1945	            (initial_state, _init_rad(phys0), phys0),
  1946	            jnp.arange(n_steps),
  1947	        )
  1948	        if return_phys_state:
  1949	            return final_state, final_ps
  1950	        return final_state
  1951	
  1952	    if phys_state_in is not None or return_phys_state:
  1953	        raise ValueError(
  1954	            "spectral_rollout: phys_state_in/return_phys_state need a "
  1955	            "physics_fn carrying the with_phys_state/init_phys_state "
  1956	            "markers (the classical split-rad factory attaches them); this "
  1957	            "physics_fn has none, so the state would be silently ignored.")
  1958	
  1959	    step_fn_ckpt = jax.checkpoint(
  1960	        step_fn_gated,
  1961	        prevent_cse=True,
  1962	        policy=jax.checkpoint_policies.nothing_saveable,
  1963	    )
  1964	
  1965	    (final_state, _), _ = jax.lax.scan(
  1966	        step_fn_ckpt,
  1967	        (initial_state, _init_rad()),
  1968	        jnp.arange(n_steps),
packages/atmosphere/legoesm/atmosphere/forcing/sam_case_forcing.py:248:    +2.8 K/km (32-47 km), isothermal 270.65 K (47-51 km). Good to ~80 km; the
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:538:    #     DEFAULT (IRAIN=0): PRC=1350·qc^2.47·(Nc[#/cm³])^-1.79,
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:570:    # water (KK2000 PRC ∝ q_c^2.47), so evaluating them on the GRID-MEAN q_c
packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py:575:    # enhancement cf^(1−2.47)=cf^−1.47 (autoconv) / cf^−1.30 (accretion).  cf is
packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py:762:#   PRC = 1350·q_c^2.47·N_c[#/cm^3]^-1.79   (autoconversion, kg/kg/s)
packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py:766:_KK2000_AUTOCONV_QC_EXPONENT = 2.47
packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py:821:def autoconversion_kk2000(q_c, N_c_eff, rho, dt):
packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py:825:        PRC = 1350 · q_c^2.47 · (N_c[#/cm³])^−1.79      [kg/kg/s]
packages/atmosphere/legoesm/atmosphere/physics/microphysics/_warm_rain.py:829:    there is no mean-mass sigmoid onset — the ``q_c^2.47`` power gives a
packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/e3sm_cam.py:1225:    0.73048, 0.9559475, 1.244795, 1.61285, 2.079325, 2.667425, 3.404875,
packages/atmosphere/legoesm/atmosphere/physics/gravity_wave_drag/e3sm_cam.py:1226:    4.324575, 5.4654, 6.87285, 8.599725, 10.70705, 13.26475, 16.35175,
packages/atmosphere/legoesm/atmosphere/physics/convection/_ifs_test_ascent.py:543:        # (cubasen.F90:472-478); PQSEN is a specific humidity.
packages/atmosphere/legoesm/atmosphere/physics/convection/config.py:95:            "delta_shallow": {"units": "1/m", "bounds": (2.475e-05, 0.000225), "tunable_tier": 2, "transform": "sigmoid", "category": "detrainment", "reference": "Bechtold et al. (2008) IFS Cy49r1", "shape": None},
packages/atmosphere/legoesm/atmosphere/physics/turbulence/turbulence_les_tuned.yaml:31:atm.turb.CLUBBParams.C_uu_shr: 0.294731913
rg: packages/ml/legoesm/training/wb*: No such file or directory (os error 2)
rg: scripts/train*: No such file or directory (os error 2)

exec
/usr/bin/bash -lc "rg -n 'def make_classical|param_fixed|make_physics\\(' packages/ml/legoesm/ml/physics* packages/ml/legoesm/training --glob '*.py' | head -65; rg -n '643[ .]|643 GiB|643GiB' . --glob '"'!**/reviews/**'"' --glob '"'!*.ipynb'"' --glob '"'!*.csv'"' --glob '"'!*.json'"' --glob '"'!*.nc'"' --glob '"'!*.svg'"' --glob '"'!*.lock'"' --glob '"'!*.dat'"' --glob '"'!*.txt'"' --glob '"'!*.log'"' --glob '"'!*.html'"' --max-count 2 | head -15; nl -ba packages/atmosphere/legoesm/atmosphere/physics/combined.py | sed -n '1126,1158p'; nl -ba packages/coupler/legoesm/driver/physics_pipeline.py | sed -n '4120,4138p'; rg --files tests | rg '(liquid_partition|macmic|clubb_liquid)'" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 725ms:
packages/ml/legoesm/training/scale_build.py:263:    "param_fixed", "param_init",
packages/ml/legoesm/training/scale_build.py:278:def _pinned_trainable_names(param_fixed: dict) -> set:
packages/ml/legoesm/training/scale_build.py:289:    return {f"{key}.{field}" for key, fields in param_fixed.items()
packages/ml/legoesm/training/scale_build.py:546:        _param_fixed = {k: dict(v) for k, v in
packages/ml/legoesm/training/scale_build.py:547:                        (_cl.get("param_fixed", {}) or {}).items()}
packages/ml/legoesm/training/scale_build.py:589:                    | _pinned_trainable_names(_param_fixed))),
packages/ml/legoesm/training/scale_build.py:648:                param_overrides=(_param_fixed or None),
packages/ml/legoesm/training/aimip_params.py:196:# (make_aimip_classical_spectral_physics -> make_physics(sfc_*_override=...) ->
packages/ml/legoesm/training/aimip_params.py:499:        baselines) to ``make_physics(sfc_albedo_override=, sfc_emissivity_override=)``
packages/ml/legoesm/training/aimip_params.py:1565:        combined_raw = make_physics(
packages/ml/legoesm/training/aimip_params.py:1647:    non_rad_raw = make_physics(non_rad_cfg, model_type="spectral_pe", dt=dt)
packages/ml/legoesm/training/neural_gcm_spectral.py:948:    raw_fn = make_physics(cfg, model_type="spectral_pe", dt=dt)
packages/ml/legoesm/training/neural_gcm_spectral.py:1067:    raw_fn = make_physics(physics_config, model_type="spectral_pe", dt=dt,
./packages/ocean/legoesm/ocean/dynamics/ocean_pe_mpas.py:325:    # planetary Coriolis — see comment at line ~643.
./data/les_cases/ARM9707/rad:1268:    325.643      -0.219389E-05
./data/les_cases/ARM9707/rad:1560:    198.643       0.900201E-06
./packages/ocean/legoesm/ocean/fidelity/time_levels.py:354:    # (the ln_bt_fw=F Kmm branch at :634-643 is a SEPARATE array section, not
./data/les_cases/DYCOMS_RF02/snd:316: 1565.000 -999.000  304.166    2.643    9.730   -0.236
./data/les_cases/DYCOMS_RF02/snd:637: 1565.000 -999.000  304.166    2.643    9.730   -0.236
./data/les_cases/ARM9707/lsf:154:    170.9790              18   963.9643       day,levels, pres0
./data/les_cases/ARM9707/lsf:626: 11484.643   215.000   0.28626E-04   0.10125E-08     6.613     5.235     0.000
./data/les_cases/DYCOMS_RF02/snd_orig:316: 1565.000 -999.000  304.166    2.643    9.630   -0.675
./data/les_cases/DYCOMS_RF02/snd_orig:637: 1565.000 -999.000  304.166    2.643    9.630   -0.675
./data/les_cases/ARM9707/snd:154:    170.9790              18   963.9643     day,levels,pres0
./data/les_cases/ARM9707/snd:156:   465.510   915.000   308.288    14.643     1.870    10.831
./data/les_cases/DYCOMS_RF02/snd_rev:316: 1565.000 -999.000  304.166    2.643    9.730   -0.236
./data/les_cases/DYCOMS_RF02/snd_rev:637: 1565.000 -999.000  304.166    2.643    9.730   -0.236
./data/les_cases/TOGA/lsf:1344:  -999.000   625.000  -0.15768E-04  -0.19775E-07    9.7643    1.8236    0.0052
  1126	        acc = None
  1127	        state_m = state
  1128	        ps_m = phys_state
  1129	        if groups[0] == "once" and sel["once"]:
  1130	            acc = _add_scaled(acc, _run(sel["once"], state, phys_state), 1.0)
  1131	        if sel["pre"]:
  1132	            r_pre = _run(sel["pre"], state, phys_state)
  1133	            acc = _add_scaled(acc, r_pre, 1.0)
  1134	            state_m = _advance_state(state_m, r_pre[0], r_pre[1], r_pre[2],
  1135	                                     r_pre[3], r_pre[5], dt)
  1136	            # CAM order: clubb_tend_cam reads THIS step's convect_deep_tend
  1137	            # output (cmfmc, dlf), so the convective carries are visible to
  1138	            # the sub-cycle (the N=1 split hands them over one step late).
  1139	            ps_m = _carry_within_call(ps_m, r_pre[6])
  1140	        # Each sub-cycled module is applied (state AND carry) before the next
  1141	        # one runs: CAM physics_update after clubb_tend_cam, then
  1142	        # microp_driver_tend on the updated state (physpkg.F90:2097-2101).
  1143	        for _i in range(_n_macmic):
  1144	            for _fr in sel["sub"]:
  1145	                r_sub = _run([_fr], state_m, ps_m)
  1146	                state_m = _advance_state(state_m, r_sub[0], r_sub[1], r_sub[2],
  1147	                                         r_sub[3], r_sub[5], _dt_sub)
  1148	                ps_m = _carry_within_call(ps_m, r_sub[6])
  1149	                acc = _add_scaled(acc, r_sub, 1.0 / _n_macmic)
  1150	        if groups[0] != "once" and sel["once"]:
  1151	            acc = _add_scaled(acc, _run(sel["once"], state, phys_state), 1.0)
  1152	        return acc
  1153	
  1154	    def physics_fn(state, grid, sigma_coord, phys_state=None, forcing=None):
  1155	        has_v = state.v is not None
  1156	
  1157	        # ---- Held-radiation sub-cycle variant (need_rad=False) ----
  1158	        # Skip the RRTMGP/gray solve; add the cached heating from the most
  4120	    # Second half of the CLUBB liquid partition; see the MPAS call site.
  4121	    _liq_closure = bool(config._liquid_partition_resolved())
  4122	    # Called UNCONDITIONALLY.  The clearing assignment inside the helper is
  4123	    # what stops a microphysics override that already carries
  4124	    # liquid_from_closure=True from reaching a built model with no liquid
  4125	    # source; gating the call on the other flags being set left exactly that
  4126	    # leak standing (GLM).  All-None / all-False is a no-op on every other
  4127	    # knob, so this is free.
  4128	    from legoesm.atmosphere.physics.microphysics.config import (
  4129	        apply_microphysics_experiment_flags,
  4130	    )
  4131	    micro_config = apply_microphysics_experiment_flags(
  4132	        micro_config, scheme,
  4133	        hard_sat_adjust_threshold=_hs_thr,
  4134	        hard_sat_max_heating_K=_hs_cap,
  4135	        homogeneous_ice_nucleation=_hom_nuc,
  4136	        liquid_from_closure=_liq_closure,
  4137	    )
  4138	
tests/unit/test_physics_macmic.py
tests/unit/test_diag_clubb_liquid_handoff.py
tests/unit/test_clubb_liquid_partition.py

exec
/usr/bin/bash -lc "rg -n 'liquid|param_overrides|morrison_flavor|warm_rain_scheme|make_physics|split_rad' packages/ml/legoesm/training/aimip_params.py | tail -40; nl -ba packages/ml/legoesm/training/aimip_params.py | sed -n '1545,1578p;1634,1652p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/microphysics/config.py | sed -n '537,562p;878,928p'; nl -ba packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py | sed -n '266,285p'; rg -n '"'^def test_'"' tests/unit/test_clubb_liquid_partition.py tests/unit/test_physics_macmic.py | head -40; nl -ba packages/coupler/legoesm/driver/model_driver.py | sed -n '9818,9846p'; nl -ba packages/ml/legoesm/training/scale_build.py | sed -n '625,652p'" in /work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6
 succeeded in 420ms:
11:mirrors :func:`legoesm.training.neural_gcm_spectral.make_physics_params_spectral_physics`
13:Xu-Randall + gray radiation) through ``combined.make_physics`` with
196:# (make_aimip_classical_spectral_physics -> make_physics(sfc_*_override=...) ->
499:        baselines) to ``make_physics(sfc_albedo_override=, sfc_emissivity_override=)``
1064:    ``core.param_overrides.apply_param_overrides``, which raises on an unknown
1078:    from legoesm.core.param_overrides import apply_param_overrides
1108:                node = apply_param_overrides(node, overrides[key])
1136:    param_overrides: dict | None = None,
1164:    split_rad: bool = False,
1173:    configs from them, then dispatches through ``combined.make_physics``
1192:    Mirrors the role of :func:`make_physics_params_spectral_physics`
1222:        for key, fields in (param_overrides or {}).items():
1224:        param_overrides = merged
1226:    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
1295:    # (make_physics -> make_radiation_physics -> _call_radiation_backend ->
1545:    # Applied HERE, after the tree is assembled and BEFORE make_physics, so the
1548:    if param_overrides:
1550:            physics_config, param_overrides)
1561:    # physics including radiation.  Returned when ``split_rad=False``
1564:    if not split_rad:
1565:        combined_raw = make_physics(
1616:    # (``make_physics`` -> ``_make_spectral_pe_combined``) does NOT
1647:    non_rad_raw = make_physics(non_rad_cfg, model_type="spectral_pe", dt=dt)
  1545	    # Applied HERE, after the tree is assembled and BEFORE make_physics, so the
  1546	    # spliced leaves are traced inside the loss (SegmentForcing doctrine) and
  1547	    # both the combined and rad-split paths below inherit them.
  1548	    if param_overrides:
  1549	        physics_config = _splice_scheme_overrides(
  1550	            physics_config, param_overrides)
  1551	        # The rad-split branch rebuilds a PhysicsConfig from the ORIGINAL
  1552	        # sub-configs, so re-read the spliced ones rather than letting that
  1553	        # path silently keep untrained values.
  1554	        rad_cfg = physics_config.radiation
  1555	        conv_cfg = physics_config.convection
  1556	        turb_cfg = physics_config.turbulence
  1557	        micro_cfg = physics_config.microphysics
  1558	        gwd_cfg = physics_config.gravity_wave_drag
  1559	
  1560	    # Combined (legacy) path: a single callable computes every-step
  1561	    # physics including radiation.  Returned when ``split_rad=False``
  1562	    # so the caller can dispatch through the simple
  1563	    # ``spectral_rollout(state, physics_fn, ...)`` branch.
  1564	    if not split_rad:
  1565	        combined_raw = make_physics(
  1566	            physics_config, model_type="spectral_pe", dt=dt,
  1567	            sfc_albedo_override=_sfc_albedo_override,
  1568	            sfc_emissivity_override=_sfc_emissivity_override,
  1569	        )
  1570	
  1571	        def combined_fn(state, grid_, sigma_coord):
  1572	            result = combined_raw(state, grid_, sigma_coord)
  1573	            return result[0] if isinstance(result, tuple) else result
  1574	
  1575	        def combined_fn_with_phys_state(state, grid_, sigma_coord, phys_state,
  1576	                                        forcing=None):
  1577	            # Same stateful contract as the split-rad entry below (codex P1:
  1578	            # the non-split path must not stay silently memoryless).
  1634	    )
  1635	    # The split builds radiation and the non-radiative physics SEPARATELY, so
  1636	    # neither sees the other and the combined wrapper's producer gate never
  1637	    # runs: cam6_clubb + a non-CLUBB closure would build with the routing on
  1638	    # and read a cloud fraction nobody writes (zero => clouds cleared).  Same
  1639	    # gate as combined.py, applied to the RESOLVED pair before splitting.
  1640	    if rad_cfg.use_clubb_cloud_fraction and turb_cfg.scheme != "clubb":
  1641	        raise ValueError(
  1642	            "RadiationConfig.use_clubb_cloud_fraction=True requires a "
  1643	            "cloud-fraction-producing turbulence closure "
  1644	            "(turbulence.scheme='clubb', diagnostic or prognostic); got "
  1645	            f"turbulence.scheme={turb_cfg.scheme!r} (cloud_scheme={cloud_scheme!r})."
  1646	        )
  1647	    non_rad_raw = make_physics(non_rad_cfg, model_type="spectral_pe", dt=dt)
  1648	    rad_only_raw = make_radiation_physics(
  1649	        rad_cfg, "spectral_pe",
  1650	        sfc_albedo_override=_sfc_albedo_override,
  1651	        sfc_emissivity_override=_sfc_emissivity_override,
  1652	        use_clubb_cloud_fraction=rad_cfg.use_clubb_cloud_fraction,
   537	    #   "kk2000" (default) = Khairoutdinov-Kogan 2000, the SAM M2005
   538	    #     DEFAULT (IRAIN=0): PRC=1350·qc^2.47·(Nc[#/cm³])^-1.79,
   539	    #     PRA=67·(qc·qr)^1.15. Faithful to the gSAM oracle.
   540	    #   "seifert_beheng" = the legacy simplified SB proxy (k_au·q_c^2·onset
   541	    #     sigmoid autoconv + bilinear k_ac·rho·q_c·q_r accretion). NOT the
   542	    #     published universal functions — only the au/x_* number closure is
   543	    #     SB-faithful.
   544	    #   "seifert_beheng_sb2001" = the PUBLISHED SB2001 universal functions —
   545	    #     the gSAM IRAIN=1 MASS closures (module_mp_graupel.f90:1835-1844,
   546	    #     :1960-1962): phi_au=600·tau^0.68·(1−tau^0.68)^3 with the q_c^4·N_c^-2
   547	    #     rate, phi_ac=(tau/(tau+5e-4))^4 with the fixed 5.78 kernel, plus the
   548	    #     au/x_* rain-number source. Faithful for a SUPPLIED nu; the default
   549	    #     fixed nu (0.4315) is an approximation of gSAM's spatially-diagnosed
   550	    #     pgam, and the SB2001-specific cloud-NUMBER sinks are not represented.
   551	    warm_rain_scheme: str = "kk2000"
   552	    predict_Nc: bool = False         # SAM M2005 dopredictNc. False (SAM DEFAULT) =
   553	                                     # SPECIFIED constant droplet number Nc_0: the
   554	                                     # size distribution uses Nc_0 and cloud number is
   555	                                     # NOT evolved (dN_c/dt=0). True = prognostic Nc,
   556	                                     # which needs SAM's droplet-activation source
   557	                                     # (not yet ported) — leaving it False matches SAM
   558	                                     # and avoids the sink-only Nc<0 drift.
   559	    # Aerosol-CCN coupling (AMIP): with predict_Nc=False, the physics
   560	    # pipeline fills ``hydrometeors.N_c`` with a per-column SPECIFIED
   561	    # droplet number diagnosed from the prescribed aerosol optical depth
   562	    # (Andreae 2009 AOT–CCN inversion, ``aerosol_activation.ccn_from_aod``)
   878	    # physically-based Abdul-Razzak & Ghan (2000) modal activation
   879	    # (``arg_activation.activated_nc_field``). Appended LAST (nested config,
   880	    # non-float) so positional construction and the param spec are unaffected.
   881	    activation: ActivationConfig = ActivationConfig()
   882	    # --- Hard (iterated) saturation-adjustment guard (opt-in) ---
   883	    # Iterated (bracketed-bisection) saturation adjustment that lands q_v ON liquid
   884	    # saturation curve where q_v > hard_sat_adjust_threshold * q_sat, draining
   885	    # local super-saturation pools the smooth sigmoid path cannot (conserving
   886	    # c_pd*T + L_v*q_v exactly).  Default OFF => byte-identical to the smooth
   887	    # path.  Threaded from ExperimentConfig via
   888	    # ``apply_microphysics_experiment_flags``.  See
   889	    # ``_warm_rain.saturation_adjustment``.
   890	    hard_saturation_adjustment: bool = False
   891	    hard_sat_adjust_threshold: float = 1.1      # RH trigger q_v > thr*q_sat [-]
   892	    hard_sat_max_heating_K: float = 5.0         # per-step latent-heating cap [K]
   893	
   894	    # --- Who supplies the cloud liquid: this scheme, or the turbulence closure
   895	    # Default False keeps this scheme's saturation adjustment as the liquid
   896	    # SOURCE, which is what every run before this field did.
   897	    #
   898	    # True says an assumed-PDF closure (CLUBB) already diagnosed the layer's
   899	    # cloud liquid and handed it to the host, so the saturation adjustment here
   900	    # is switched OFF ENTIRELY -- both signs.  This is the CAM6 arrangement, not
   901	    # an invention: micro_mg2_0.F90:2688-2730 carries a residual
   902	    # "remove any excess over-saturation" block, it is gated on
   903	    # ``allow_sed_supersat`` at :2700, and micro_mg_cam.F90:668-672 sets that
   904	    # flag ``.false.`` whenever ``do_clubb_sgs``.  Enumerating every write to
   905	    # ``qctend`` in MG2 (:2634 ice melt, :2680 homogeneous freezing -- a sink,
   906	    # :2718 the gated residual) leaves MG2 with NO vapour-to-liquid
   907	    # condensation at all in a CLUBB configuration.
   908	    #
   909	    # POSITIVE BRANCH ONLY.  An earlier version said "both signs"; that was
   910	    # wrong twice.  CAM's block is guarded by ``qtmp > qvn``
   911	    # (micro_mg2_0.F90:2700), so it fires on positive supersaturation and has
   912	    # no evaporation branch to switch off.  And our default
   913	    # ``wbf_scheme="emergent"`` has no explicit Bergeron rate: the mixed-phase
   914	    # cloud-water sink IS that negative branch.  Every genuine sink
   915	    # (autoconversion, accretion, the emergent WBF, riming, sedimentation) is
   916	    # untouched.
   917	    #
   918	    # WHY IT MATTERS: the adjustment removes the whole supersaturation on every
   919	    # call, so running microphysics N times inside one physics step adjusts N
   920	    # times.  A 3/6/15 sub-step sweep moved band liquid monotonically by 21%,
   921	    # which is a step-count sensitivity a liquid SOURCE should not have.
   922	    #
   923	    # Turning this on without the closure actually delivering liquid removes the
   924	    # model's only liquid source; ``ExperimentConfig.validate_strict`` refuses
   925	    # that pairing rather than letting it run.
   926	    liquid_from_closure: bool = False
   927	
   928	    # --- IFS/SAM homogeneous-freezing ice-supersaturation allowance ---
   266	    sharpness = config.saturation_sharpness
   267	
   268	    N_c_eff = effective_Nc(
   269	        N_c, config.Nc_0,
   270	        predict_Nc=getattr(config, "predict_Nc", False),
   271	        nc_specified_field=getattr(config, "nc_from_aerosol", False),
   272	    )
   273	
   274	    # === WARM RAIN (shared Seifert-Beheng helpers) ===
   275	    # Pass ``q_c`` so the evaporation branch (negative ``condensation``)
   276	    # is donor-clamped: evaporation cannot drive ``q_c`` below zero in
   277	    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
   278	    condensation, q_sat = saturation_adjustment(
   279	        T, q_v, p_full, dt, sharpness, q_c=q_c,
   280	        hard_adjust=config.hard_saturation_adjustment,
   281	        hard_threshold=config.hard_sat_adjust_threshold,
   282	        hard_max_heating_K=config.hard_sat_max_heating_K,
   283	    )
   284	    # CAM6 arrangement: when the turbulence closure already diagnosed this
   285	    # layer's cloud liquid and handed it to the host, MG2 carries no
tests/unit/test_clubb_liquid_partition.py:77:def test_default_is_off_and_emits_no_liquid():
tests/unit/test_clubb_liquid_partition.py:84:def test_off_path_is_bit_identical_to_the_historical_bridge():
tests/unit/test_clubb_liquid_partition.py:105:def test_total_water_is_conserved_by_the_partition():
tests/unit/test_clubb_liquid_partition.py:126:def test_liquid_is_replaced_not_accumulated():
tests/unit/test_clubb_liquid_partition.py:169:def test_vapour_stays_non_negative():
tests/unit/test_clubb_liquid_partition.py:178:def test_moist_static_energy_round_trip_is_exact_without_cloud():
tests/unit/test_clubb_liquid_partition.py:200:def test_latent_heat_accompanies_the_condensate():
tests/unit/test_clubb_liquid_partition.py:229:def test_sub_cycle_carries_the_liquid_between_steps():
tests/unit/test_clubb_liquid_partition.py:289:def test_mismatched_pairing_raises_both_ways():
tests/unit/test_clubb_liquid_partition.py:308:def test_non_mpas_lanes_refuse_the_lever():
tests/unit/test_clubb_liquid_partition.py:322:def test_partition_uses_the_grid_mean_liquid_not_the_in_cloud_one():
tests/unit/test_clubb_liquid_partition.py:376:def test_partition_requires_the_sequential_macro_micro_subcycle():
tests/unit/test_clubb_liquid_partition.py:401:def test_partition_refuses_the_diagnostic_condensate_floor():
tests/unit/test_clubb_liquid_partition.py:428:def test_the_host_applies_the_subcycled_modules_in_sequence():
tests/unit/test_clubb_liquid_partition.py:470:def test_an_override_cannot_smuggle_the_lever_past_the_ordering_guard():
tests/unit/test_clubb_liquid_partition.py:500:def test_condensate_loading_is_carried_in_the_buoyancy():
tests/unit/test_clubb_liquid_partition.py:537:def test_mpas_lane_refuses_a_state_missing_either_tracer():
tests/unit/test_clubb_liquid_partition.py:560:def test_public_wrapper_refuses_liquid_with_the_lever_off():
tests/unit/test_clubb_liquid_partition.py:574:def test_an_override_cannot_smuggle_the_lever_past_the_floor_guard():
tests/unit/test_clubb_liquid_partition.py:628:def test_liquid_from_closure_defaults_off():
tests/unit/test_clubb_liquid_partition.py:634:def test_liquid_from_closure_removes_the_condensation_source():
tests/unit/test_clubb_liquid_partition.py:657:def test_liquid_from_closure_leaves_every_sink_untouched():
tests/unit/test_clubb_liquid_partition.py:684:def test_liquid_from_closure_rejected_on_a_scheme_without_it():
tests/unit/test_clubb_liquid_partition.py:695:def test_helper_sets_AND_clears_the_flag():
tests/unit/test_clubb_liquid_partition.py:717:def test_mixed_phase_liquid_sink_survives_the_gate():
tests/unit/test_clubb_liquid_partition.py:773:def test_gate_removes_the_latent_heating_with_the_water():
tests/unit/test_clubb_liquid_partition.py:797:def test_gate_also_covers_the_hard_adjustment_path():
tests/unit/test_clubb_liquid_partition.py:828:def test_microphysics_half_alone_is_refused_at_the_factory():
tests/unit/test_clubb_liquid_partition.py:852:def test_scalar_threading_does_not_undo_the_gate():
tests/unit/test_physics_macmic.py:119:def test_n1_default_is_the_parallel_split():
tests/unit/test_physics_macmic.py:136:def test_macmic_matches_the_sequential_reference_and_is_not_a_rescaling():
tests/unit/test_physics_macmic.py:163:def test_jit_parity_and_finite_gradient():
tests/unit/test_physics_macmic.py:189:def test_held_radiation_cache_and_ledger_route_through_the_loop():
tests/unit/test_physics_macmic.py:230:def test_per_call_inputs_survive_the_sub_cycle_and_clear_at_the_end(monkeypatch):
tests/unit/test_physics_macmic.py:266:def test_validation_cli_and_amip_round_trip():
tests/unit/test_physics_macmic.py:308:def test_cam6_deck_reaches_the_config_through_the_yaml_route():
  9818	                cfg.microphysics: apply_microphysics_experiment_flags(
  9819	                    _msub, cfg.microphysics,
  9820	                    nc_from_aerosol=cfg.nc_from_aerosol,
  9821	                    subgrid_autoconversion=cfg.subgrid_autoconversion,
  9822	                    homogeneous_ice_nucleation=getattr(
  9823	                        cfg, "homogeneous_ice_nucleation", False),
  9824	                    # The OTHER half of the CLUBB liquid partition: once the
  9825	                    # closure hands its diagnosed liquid to the host, the
  9826	                    # microphysics must stop manufacturing its own, exactly as
  9827	                    # CAM6 switches MG2's residual adjustment off under CLUBB
  9828	                    # (micro_mg_cam.F90:668-672).  One flag drives both halves
  9829	                    # so they can never be enabled apart -- the microphysics
  9830	                    # half alone would leave the model no liquid source at all.
  9831	                    liquid_from_closure=cfg._liquid_partition_resolved(),
  9832	                    # NOTE: hard_saturation_adjustment is INTENTIONALLY NOT
  9833	                    # threaded in-scheme on the MPAS path.  The integration
  9834	                    # trial showed the in-scheme placement cannot correct the
  9835	                    # MPAS dycore's per-step vertical vapour-transport spike
  9836	                    # within the dt window (blew up at day 24), while the same
  9837	                    # adjustment applied POST-STEP on the final state is the
  9838	                    # proven-stable intervention (40/40 + multi-month pilots).
  9839	                    # So on MPAS the flag is applied post-step below; the
  9840	                    # coupled path (physics_pipeline) still applies it in-scheme.
  9841	                    # The float trigger/heating-cap OVERRIDES are threaded here
  9842	                    # so the post-step reads below pick them up (the post-step
  9843	                    # drain reads the scheme sub-config, not ExperimentConfig).
  9844	                    hard_sat_adjust_threshold=getattr(
  9845	                        cfg, "hard_sat_adjust_threshold", None),
  9846	                    hard_sat_max_heating_K=getattr(
   625	        # holds one block's activations instead of every column's.  Needed
   626	        # when max-random overlap expands the solver's column axis by the
   627	        # sub-column count (8 sub-columns at T63/L32 asks for 490 GiB
   628	        # otherwise).  0 = off, and the value must divide the solver's column
   629	        # count (n_sub * ncol when sub-columns are on).
   630	        _col_chunk = int(_cl.get("rrtmgp_column_chunk_size", 0))
   631	        if _col_chunk < 0:
   632	            raise ValueError(
   633	                "classical.rrtmgp_column_chunk_size must be >= 0, got "
   634	                f"{_col_chunk}; 0 disables column chunking.")
   635	        _gpt_batch = int(_cl.get("rrtmgp_gpoint_batch_size", 16))
   636	        if _gpt_batch < 1:
   637	            raise ValueError(
   638	                "classical.rrtmgp_gpoint_batch_size must be >= 1, got "
   639	                f"{_gpt_batch}; use the scan path by setting it to 1 rather "
   640	                "than 0 or a negative value.")
   641	
   642	        def make_physics_fn(p):
   643	            return make_aimip_classical_spectral_physics(
   644	                p, grid, dt, radiation=_radiation, split_rad=True,
   645	                rad_update_interval_steps=_rad_interval,
   646	                rrtmgp_gpoint_batch_size=_gpt_batch,
   647	                rrtmgp_column_chunk_size=_col_chunk,
   648	                param_overrides=(_param_fixed or None),
   649	                orbital_insolation=bool(
   650	                    _cl.get("orbital_insolation", False)),
   651	                convective_rain_to_surface=bool(
   652	                    _cl.get("convective_rain_to_surface", False)),

codex
HOLD

1. **C1 omits essential parts of the exchange.** MPAS requires **both `q_v` and `q_c`**, passes existing liquid into CLUBB, rejects missing liquid output, then routes the tendency. The cited lines 847/1022 cover validation/input; output routing is at [integration.py:1080](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1080). Merely adding output routing fails: [clubb.py:6330](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6330) requires host liquid to initialize total water and liquid-water temperature. Piece A is useful scaffolding, but has no production effect while partition remains refused.

2. **The claim misses the microphysics half.** Production also enables `MorrisonConfig.liquid_from_closure` at [model_driver.py:9831](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/coupler/legoesm/driver/model_driver.py:9831). Without it, Morrison still produces liquid independently. Setting `hard_saturation_adjustment: false` does **not** disable condensation; the separate gate is [morrison.py:302](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:302). C1+C2 alone therefore do not establish production consistency.

3. **Ordering is necessary; the proposed implementation is not proven necessary.** The cited `combined.py:251`, `:281`, `:606`, `:1385` and turbulence `integration.py:545` all check out. Parallel evaluation is explicit at [combined.py:1497](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/combined.py:1497). But updating **only liquid** leaves microphysics with stale vapour and temperature, despite CLUBB changing all three ([clubb.py:6487](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6487)). A grid-space sequential implementation could avoid repeated transforms; that is a candidate requiring validation, not an established equivalent. The existing reference advances full state **and carry**, with convection first ([combined.py:1131](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/combined.py:1131)).

4. **Porting the combined function alone cannot activate this deck.** The active WB builder selects split radiation at [scale_build.py:644](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/scale_build.py:644); its non-radiative factory supplies no subcycle count ([aimip_params.py:1647](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/aimip_params.py:1647)). Configuration plumbing is another required piece. C3 also overstates the deck changes: only the Morrison hard-adjustment row exists ([spectral_t63_cam6.yaml:285](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/config/wb/campaign/spectral_t63_cam6.yaml:285)); there is no explicit ice-curve row, and this in-scheme switch is distinct from MPAS’s post-step switch.

5. **The memory argument is unverified.** The claim supplies no artifact for 643 GiB. Step checkpointing is confirmed at [neural_gcm_spectral.py:1938](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/neural_gcm_spectral.py:1938), but it does not establish peak memory or compile cost for a proposed subcycle. Measure identical T63/L32 forward-and-gradient workloads, recording compile time, compiler memory estimates, runtime peak and step time.

6. **Before shipping, demonstrate the complete contract.** Require condensation **and evaporation** cases; missing-either-tracer rejection; water budgets including surface flux and precipitation; thermodynamic consistency; sequential-reference agreement; JIT and gradient checks; and unchanged MPAS/hydrostatic behavior. Include planted failures for dropped liquid routing, stale microphysics inputs and disabled closure pairing. Conservation alone cannot validate ordering. This review verified source; runtime behavior and performance remain unmeasured.
tokens used
76,328
HOLD

1. **C1 omits essential parts of the exchange.** MPAS requires **both `q_v` and `q_c`**, passes existing liquid into CLUBB, rejects missing liquid output, then routes the tendency. The cited lines 847/1022 cover validation/input; output routing is at [integration.py:1080](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/turbulence/integration.py:1080). Merely adding output routing fails: [clubb.py:6330](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6330) requires host liquid to initialize total water and liquid-water temperature. Piece A is useful scaffolding, but has no production effect while partition remains refused.

2. **The claim misses the microphysics half.** Production also enables `MorrisonConfig.liquid_from_closure` at [model_driver.py:9831](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/coupler/legoesm/driver/model_driver.py:9831). Without it, Morrison still produces liquid independently. Setting `hard_saturation_adjustment: false` does **not** disable condensation; the separate gate is [morrison.py:302](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/microphysics/morrison.py:302). C1+C2 alone therefore do not establish production consistency.

3. **Ordering is necessary; the proposed implementation is not proven necessary.** The cited `combined.py:251`, `:281`, `:606`, `:1385` and turbulence `integration.py:545` all check out. Parallel evaluation is explicit at [combined.py:1497](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/combined.py:1497). But updating **only liquid** leaves microphysics with stale vapour and temperature, despite CLUBB changing all three ([clubb.py:6487](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/turbulence/clubb.py:6487)). A grid-space sequential implementation could avoid repeated transforms; that is a candidate requiring validation, not an established equivalent. The existing reference advances full state **and carry**, with convection first ([combined.py:1131](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/atmosphere/legoesm/atmosphere/physics/combined.py:1131)).

4. **Porting the combined function alone cannot activate this deck.** The active WB builder selects split radiation at [scale_build.py:644](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/scale_build.py:644); its non-radiative factory supplies no subcycle count ([aimip_params.py:1647](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/aimip_params.py:1647)). Configuration plumbing is another required piece. C3 also overstates the deck changes: only the Morrison hard-adjustment row exists ([spectral_t63_cam6.yaml:285](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/config/wb/campaign/spectral_t63_cam6.yaml:285)); there is no explicit ice-curve row, and this in-scheme switch is distinct from MPAS’s post-step switch.

5. **The memory argument is unverified.** The claim supplies no artifact for 643 GiB. Step checkpointing is confirmed at [neural_gcm_spectral.py:1938](/work/bd1083/b309178/diffESM/legoesm_pg/wt_wbcam6/packages/ml/legoesm/training/neural_gcm_spectral.py:1938), but it does not establish peak memory or compile cost for a proposed subcycle. Measure identical T63/L32 forward-and-gradient workloads, recording compile time, compiler memory estimates, runtime peak and step time.

6. **Before shipping, demonstrate the complete contract.** Require condensation **and evaporation** cases; missing-either-tracer rejection; water budgets including surface flux and precipitation; thermodynamic consistency; sequential-reference agreement; JIT and gradient checks; and unchanged MPAS/hydrostatic behavior. Include planted failures for dropped liquid routing, stale microphysics inputs and disabled closure pairing. Conservation alone cannot validate ordering. This review verified source; runtime behavior and performance remain unmeasured.
