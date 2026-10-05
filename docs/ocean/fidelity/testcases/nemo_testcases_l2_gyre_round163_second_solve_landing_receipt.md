# NEMO testcase L2 GYRE Round 163 receipt — the second continuity solve LANDS

Date: 2026-09-24

Status: **LANDED, for GYRE-zco only** — round 160's second per-stage
continuity solve is now GYRE-zco's own explicit production config choice,
under Decision 55 (note AT, the amended year gate). The review below found a
real BLOCKER first (ORCA2-zps would have silently taken the same route,
unmeasured); it is fixed twice — first with an EOS-keyed default the
coordinator then rejected as a hidden coupling between unrelated choices,
then with the shipped shape: each card that resolves the two-solve program
states its own explicit choice on its own config.

## Outcome first

The flip reproduces round 160's candidate-arm numbers to every printed digit:
stage-2 momentum right-hand side lands on round 158's oracle-`ww` ceiling, the
step-2 `u`/`v` velocity rows go DEBT → AT-BAR, the first row over the bar moves
from step 2 to step 3, and day 30 falls
`6.88819351379691829e-05` → `6.57257437477060260e-05` K. Day 240 worsens
`1.64467186475764637e-02` → `1.64483607011786798e-02` K (+9.98e-05 relative)
and day 360 worsens `1.12234508497438962e-02` → `1.12256600185513065e-02` K
(+1.97e-04 relative) — both under Decision 55's 1e-3 relative bar, so both are
**REGISTERED**, not refused. Every other Decision 43/45 criterion PASSES.
The other cards (generic NEMO-GYRE recipe, both tanks, both DINO recipes) do
not execute the route and do not move: 170 tests, identical count to round
160's own measurement of them.

**The independent review found ORCA2-zps would also have silently executed
the new route, unmeasured** (BLOCKER, closed same round — see Review). The
fix that closes it does not touch any code path GYRE-zco's own resolved
config reaches, and a direct predicate check at the final commit confirms
GYRE-zco still executes the route (`test_orca2_resolves_the_program_but_its_own_config_excludes_it`,
part of the green push-gate battery below) — so the numbers above, measured
at the pre-fix commit, are unchanged at the commit that ships. This is
argued from the code, not re-measured through a second 40-minute model
battery; see "What was re-verified vs. reused" below.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round163.md`, committed as
`987af9a71` before any round-163 measurement ran. The initial flip is
`af15e39c4`. The before arm is round 160's own already-measured base commit
`de4e0cfe35daa6e6e5282a9b192a78abffb47ada` (its ladder, month and year members
are reused unchanged, since rounds 161 and 162 both touched zero files under
`packages/` — checked again this round: `git diff --stat de4e0cfe35da..b18cfc276e39 -- packages/` shows only round 160's own private-arm commits, round 161's and
round 162's measurement/test-only commits). The candidate arm's full model
battery (ladder, day-30/240/360, day-240 decomposition, other-cards gate) was
measured at `8ec2fba50296ffe51dab7c0c960ff8c9c5b6f4f1`. The review then found
the ORCA2 BLOCKER; this round fixed it twice (`29abe4973`, an EOS-keyed
exclusion, then `3103c9862`, the coordinator-directed explicit per-card
config choice that replaced it), with a citation re-anchor after each
(`a7138998b`, `dc77a4e5b`) and the receipt trued up after each
(`b2deb94d4`, `84864a095`). The round ships at `84864a095`, the tip of this
range. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round163/`.

## The flip

Decision 55 (note AT): a NEMO statement that is (1) cited from compiled
`ppsrc`, (2) proven one-variable, and (3) takes a certified trajectory row
from DEBT to AT-BAR, LANDS even when day-240/360 T rms move by less than 1e-3
relative. Round 160 already proved all three; round 163 flips the production
default.

Three shapes, in order, the last one shipped. **First attempt** (`af15e39c4`):
`_NEMOWSRK3TestHooks.nemo_stage_momentum_wzv_split` default `False` → `True`
in `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`, so
every card resolving the program landed it. Review found this too broad (the
BLOCKER below). **Second attempt** (`29abe4973`): kept the flat default but
excluded ORCA2 by INFERRING from `eos == "nemo_teos10"`. The coordinator
rejected this: EOS says nothing about continuity, so keying one on the other
is a hidden coupling between unrelated choices, and the rule is that every
scientific choice appears in the run's config, not in a default. **Final
shape** (this commit range's last edits): `nemo_stage_momentum_wzv_split` is
now a real field on `LatLonCGridOceanConfig` (`state.py`, next to the
`wzv_call2_evaluation` field it depends on) — an EXPLICIT per-card choice.
GYRE-zco's own config branch (`nemo_testcase_recipe.py`) sets it `True`;
ORCA2-zps's branch, which specializes the identical base and resolves the
same program, sets it `False`. Neither is inferred from anything else.
`nemo_stage_momentum_wzv_executes(config, hooks=None)` reads
`config.nemo_stage_momentum_wzv_split`; if a resolving card leaves it unset
(`None`, the field's construction default), the model RAISES rather than
guessing — fail-closed for every future card, not only ORCA2. The
TEST-ONLY hook (a *different* object, `_NEMOWSRK3TestHooks`, same field
name) is unchanged from rounds 159-162: an explicit hook still always wins,
subject only to `nemo_stage_momentum_wzv_resolved`. Grepped: the only
`_NEMOWSRK3TestHooks(` call anywhere in `packages/` is the model's own
`_nemo_ws_test_hooks or _NEMOWSRK3TestHooks()` default, so the card-level
config field is the whole production mechanism. Five of this round's own
structural tests (in the round-160 test file, updated this round) cover the
opt-out hook, the ORCA2 config exclusion, and the fail-closed raise for an
unset config.

### The compiled statement, re-cited

`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360` (the raw stage
velocity, velocity indicator) and
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130` (its continuity
solve) for momentum; `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`
for the tracer re-solve, unchanged. Same citations round 160 gave; re-quoted
here rather than only referenced, per note AT item 1.

### Card census, re-verified from each card's resolved configuration

| card | momentum integrator | momentum advection | continuity solve | resolves the program | own config choice | executes |
|---|---|---|---|---|---|---|
| GYRE-zco | rk3_ws | vector_invariant | nemo_literal | **yes** | `nemo_stage_momentum_wzv_split=True` | **yes** |
| ORCA2-zps | rk3_ws | vector_invariant | nemo_literal | **yes** | `nemo_stage_momentum_wzv_split=False` | **no** |
| LOCK_EXCHANGE-zco | rk3_ws | flux_form | generic | no | — (never reached; resolved() is False) | no |
| OVERFLOW-zps | rk3_ws | flux_form | generic | no | — | no |
| NEMO-GYRE recipe | rk3_ws | vector_invariant | generic | no | — | no |
| DINO nemo_dino_kamm | euler | vector_invariant | nemo_literal | no | — | no |
| DINO nemo_dino_kamm_mlf | euler | vector_invariant | nemo_literal | no | — | no |

**ORCA2-zps is NOT a card the decision43_45 gate's own `_card_execution`
enumerates** (it never has — round 160's table omitted it too, and round
163's review is what found the omission mattered). It resolves the SAME
three base conditions GYRE-zco does by specializing the identical
`gyre_vector_ene_c2` config branch (`nemo_testcase_recipe.py`) and never
overriding `momentum_time_integrator`/`momentum_advection`/
`wzv_call2_evaluation` — checked directly against its resolved config, built
via `_model_config(whole_step_identity="orca2_vector_een_c2", ...)`, no deck
files needed. Two cards now resolve the program; each states its OWN
explicit `nemo_stage_momentum_wzv_split` choice on its resolved config —
GYRE-zco `True`, ORCA2-zps `False` — and neither choice is inferred from any
other field. Re-run this round via
`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round160_wzv_split.py::test_orca2_resolves_the_program_but_its_own_config_excludes_it`,
`::test_a_card_that_resolves_the_program_without_a_config_choice_raises`
and `::test_the_admission_gate_census_uses_the_model_s_own_predicate`
(updated: `executes_at_this_tip` asserts `["GYRE-zco"]`). The tanks, generic
recipe and both DINO recipes are out on the momentum integrator, the
momentum advection form, or the continuity-solve selector — unchanged from
round 160.

## The full Decision 43/45 gate

Before arm: `de4e0cfe35daa6e6e5282a9b192a78abffb47ada` (round 160's base,
reused). Candidate: `8ec2fba50296ffe51dab7c0c960ff8c9c5b6f4f1`. Report:
`decision43_45.json`.

| criterion | verdict |
|---|---|
| day-30 T rms decreases | **PASS** 6.888193513796918e-05 → 6.572574374770603e-05 K, factor 1.048020626 |
| first row over the bar not earlier | **PASS** step 2 (u, v) → step 3 (T, S, u, v, ssh) |
| no kt=1 at-bar row leaves | **PASS** 0 losses |
| every moved row registered | **PASS** 60 moved, 60 registered, 0 missing, 0 unexpected |
| every year row registered | **PASS** 8/8 |
| DINO measured if shared | **PASS** not executed (census) |
| all executing cards measured | **PASS** |
| day 240 T rms not worse | **FAIL, amended by Decision 55** — see below |
| day 360 T rms not worse | **FAIL, amended by Decision 55** — see below |

### Note-AT registration, the two amended rows

The harness's run-to-run floor is ~2e-10 K (round 129), quoted next to both.

| day | before (K) | after (K) | change | relative | multiples of the floor |
|---:|---:|---:|---:|---:|---:|
| 240 | 1.6446718647576464e-02 | 1.6448360701178680e-02 | +1.6421e-06 | **+9.98e-05** | ~8,210 |
| 360 | 1.1223450849743896e-02 | 1.1225660018551306e-02 | +2.2091e-06 | **+1.97e-04** | ~11,046 |

Both relative moves sit under Decision 55's 1e-3 bar. Both are thousands of
times the run-to-run floor in absolute terms — not noise — and both are
registered exactly as measured, not excused. All eight year rows, same as
round 160's (reproduced to every printed digit this round):

| day | before (K) | after (K) | relative |
|---:|---:|---:|---:|
| 30 | 6.8881935138e-05 | 6.5725743748e-05 | −4.58e-02 |
| 60 | 1.9325579917e-04 | 2.0808020004e-04 | +7.67e-02 |
| 90 | 1.8644545401e-03 | 1.8643018774e-03 | −8.19e-05 |
| 120 | 1.0500500326e-03 | 1.0484738105e-03 | −1.50e-03 |
| 180 | 3.5805134081e-03 | 3.5838854686e-03 | +9.42e-04 |
| 240 | 1.6446718648e-02 | 1.6448360701e-02 | +9.98e-05 |
| 300 | 1.3597381746e-02 | 1.3602990041e-02 | +4.13e-04 |
| 360 | 1.1223450850e-02 | 1.1225660019e-02 | +1.97e-04 |

### The certified ladder, 70 rows

`traj_comparison.json` (produced with `--trajectory-only`, the flag that
scores exactly the 70 certified trajectory rows rather than the raw walk's
954 diagnostic sub-arm probes — confirmed
`n_certified_rows_compared: 70`). 60 of 70 rows moved; 494,431 cells improved
and 114,308 worsened — identical counts to round 160's own measurement.
Largest worsening anywhere: `GYRE-zco.kt6.before.u`, `4.9400e-06`, 13,496
cells improved / 3,904 worsened. Two rows changed status, both DEBT → AT-BAR,
none the other way:

| row | cells improved | cells worsened | largest worsening (units in last place) |
|---|---:|---:|---:|
| GYRE-zco.kt2.before.u | 17,397 | 2 | 4.1633e-17 (0.1875 ulp) |
| GYRE-zco.kt2.before.v | 17,100 | 0 | 0 |

The kt=2 temperature and salinity rows do not move; they stay at the bar. The
moved-row registry (`moved_rows.tsv`, 60 rows) is byte-identical to round
160's own `moved_rows.tsv` — diffed directly, zero lines differ.

## Day-240 owner ranking in the landed arm vs production (note AB method)

`year_owners.py --decompose 240` on both year members: the landed arm
(`round163/after_year`) and the production (pre-flip) arm
(`round160/base_year`, round 160's own reused before member). Same instrument
round 122 and round 128 used.

| field / axis | landed arm (T rms, share) | production arm (T rms, share) |
|---|---|---|
| leading field (dimensionless) | v | v |
| depth 0–100 m | 3.0687e-02 K, 92.82% of ΔT² | 3.0685e-02 K, 92.82% |
| depth 100–1000 m | 7.6321e-03 K, 7.18% | 7.6294e-03 K, 7.17% |
| region west third | 2.7171e-02 K, 90.96% | 2.7169e-02 K, 90.97% |
| region south of 37.2N | 1.8441e-02 K, 93.65% | 1.8440e-02 K, 93.65% |
| wind band 15–29N | 2.6780e-02 K, 91.01% | 2.6778e-02 K, 91.01% |
| peak \|ΔT\| | −1.8864 K at j=1,i=2,k=1 (15.1 m, 16.87N) | −1.8863 K, same cell |

The spatial pattern is UNCHANGED by the landing: same leading field, same
depth band, same region, same peak cell, to three significant figures on
every share. This is a magnitude reproduction of round 122's finding, not a
new owner — the split moves the day-240 T rms by 1.6e-06 K, four orders below
the pattern it sits inside, exactly as round 162 measured for the one-step
state (no mechanism either way). This is the SPATIAL decomposition (note AB's
literal method); it is not the PROCESS-level ranking (vertical diffusion vs.
advection) that rounds 123–128, 152 and 161 used — that instrument is round
164's OPEN item, not re-run here.

## The other cards' gates

Both DINO recipes, both tanks and the tank zero-diffusion removal — same
battery round 160 ran, same count:

> 170 passed, 9 warnings in 336.80s (0:05:36)

The generic NEMO-GYRE recipe's own file is inside the push-gate battery
below. None of these cards executes the route (census above), and none moved
— consistent with the census, not merely assumed from it.

## Review

Codex is paused (per standing note), so a `code-reviewer` subagent (model
sonnet) reviewed the round's diff at commit `7ce7e9f48` (the flip, the
citation re-anchors, and the receipt as first drafted). **Verdict: DO NOT
SHIP.** One BLOCKER, closed this round; everything else checked out.

1. **BLOCKER — ORCA2-zps silently executes the new route, unmeasured, and
   the first-drafted receipt's safety claim about that was false.** The
   reviewer built ORCA2's resolved config directly
   (`_model_config(whole_step_identity="orca2_vector_een_c2", ...)`, no deck
   files) and called the real predicates:
   `nemo_stage_momentum_wzv_resolved` and (at that commit)
   `nemo_stage_momentum_wzv_executes` both `True`. ORCA2 inherits
   `momentum_time_integrator="rk3_ws"`, `momentum_advection="vector_invariant"`,
   `wzv_call2_evaluation="nemo_literal"` from the shared `gyre_vector_ene_c2`
   base it specializes and never overrides them, so the flat `True` default
   from the first attempt would have landed the new physics on ORCA2 with
   zero fidelity measurement — while the receipt's own review draft (written
   before the real review ran) claimed the opposite. **CLOSED, `29abe4973`**:
   `nemo_stage_momentum_wzv_split` is now a `bool | None` sentinel; the
   no-hook default policy additionally requires `eos == "nemo_teos10"`
   (GYRE's own EOS; ORCA2 resolves `nemo_eos80`), so ORCA2 stays on the
   pre-round-160 single shared solve by default. Non-vacuity test added
   (`test_orca2_resolves_the_program_but_is_excluded_from_the_default`):
   asserts ORCA2 resolves the program, `executes()` is `False` by default,
   and `True` with an explicit hook (the exclusion is a default, not a hard
   block — a future ORCA2-focused round can still land it once measured).
   **DECISION_NEEDED for the operator**: should ORCA2 take this route once
   measured, in its own round, or is there a reason to hold it back longer?
   Not decided here — this round only refuses to ship it silently.
   **AMENDED after the reviewer's pass, by the coordinator**: the EOS-keyed
   default policy above was itself a hidden coupling between unrelated
   choices (EOS says nothing about continuity) and violated "every
   scientific choice appears in the run's config, not in a default." Final
   shape replaces it: `nemo_stage_momentum_wzv_split` is a real field on
   `LatLonCGridOceanConfig`, set explicitly `True` on GYRE-zco's own config
   branch and explicitly `False` on ORCA2-zps's — see "The flip" above. A
   card resolving the program without setting the field raises. The
   non-vacuity test is renamed
   (`test_orca2_resolves_the_program_but_its_own_config_excludes_it`) and now
   asserts the config field, not EOS; a second test proves the raise.
2. **MEDIUM — dual-review not met.** Only one reviewer ran (codex is
   paused; GLM was not invoked). REGISTERED, not closed: recommend a GLM
   pass before this lands upstream of the operator, given the single pass
   missed a real blocker on its own first read of the (then-different)
   diff.
3. **LOW — Decision 55's override lives only in receipt prose**, not in
   `decision43_45.json` (`status: FAIL` there, unchanged by this receipt's
   registration of the two amended rows). REGISTERED for round 164 or later:
   a `decision55_override` field would make this mechanical instead of
   memorized.

What the reviewer verified and found correct (unchanged by the fix, since
none of it touches the physics or the measured numbers): the flip diff
itself minimal and well-commented; no production card-builder anywhere
constructs `_NEMOWSRK3TestHooks` with an override; every headline number in
the receipt reproduces exactly against its cited JSON (day-30/240/360 T rms
and deltas, floor multiples, cell counts, the day-240 spatial table);
`moved_rows.tsv` byte-identical to round 160's; the three compiled-NEMO
citations verified against the actual oracle build tree; the round-160 test
file correctly asserts the landed state including the opt-out; push/card-gate
pass counts verified against the actual `N passed` lines, not exit codes.

### What was re-verified vs. reused, after the fixes

Both the ORCA2 fix (`29abe4973`) and the coordinator's config-choice
redesign that replaced its EOS inference touch only HOW
`nemo_stage_momentum_wzv_executes` decides for a card with no explicit test
hook. GYRE-zco's own resolved config sets `nemo_stage_momentum_wzv_split`
explicitly `True` in every one of these shapes — its branch of the decision
never depended on ORCA2's exclusion mechanism, whether that mechanism was a
flat default, an EOS check, or (final shape) ORCA2's own separate config
choice. The SAME boolean (`True`) feeds the SAME `LatLonCGridOceanModel.step`
trace across all three shapes. This is checked directly, not just argued:
`test_orca2_resolves_the_program_but_its_own_config_excludes_it` asserts
`nemo_stage_momentum_wzv_executes(gyre_config) is True` against GYRE-zco's
actual `card.recipe.model_config`, and that assertion is part of
the green 163-test push-gate battery quoted below, run after BOTH fixes. The
full 40-minute model battery (ladder, day-30/240/360 member runs, day-240
decomposition, other-cards gate) was NOT re-run after either fix — it was
run once, at `8ec2fba50296`, and this argument is offered in place of a
second run. The receipt's physics numbers above are measured at that
commit; the shipped commit is `84864a095`, six commits later, none of which
touch GYRE-zco's code path (its own config choice, `True`, is set once and
never overridden by the redesign).

## Push gate and this round's own tests

Citation gate plus the five push-gate files plus this round's own updated
structural test file (now 29 tests, including the ORCA2 config-choice
non-vacuity test and the fail-closed raise test), run FOUR times as the
round's own edits landed, including the coordinator's config-choice
redesign — the number below is the FINAL run, on the shipped tree at
`84864a095`:

> 163 passed in 936.66s (0:15:36)

An intermediate run (after the initial flip, before the round-8 receipt's
citations were re-anchored a second time for the ORCA2 fix's own line shift)
showed the expected single failure
(`test_the_gate_runs_clean_on_the_real_receipt`, 9 unmapped citations) and is
not the number quoted as passing — the fix and the re-verification are both
in this receipt's commit range, not asserted without the failing
intermediate shown.

Citation gate detail: `audit_map()` over the FULL map (not just this file's
re-anchored entries) returns zero failing rows; `run(DEFAULT_RECEIPT,
DEFAULT_HEADING)` returns `status: PASS`, zero unmapped citations, zero
failures, zero map-audit failures, all self-tests fired.

## What changed in the citation map, mechanically

THREE rounds of edits to `ocean_model_latlon_cgrid.py` — the initial flip,
the (rejected) EOS-exclusion fix, and the final explicit-config-choice
redesign — each growing the field-default comment and the
`nemo_stage_momentum_wzv_executes` docstring/body, and the last also adding
`nemo_stage_momentum_wzv_resolved`'s docstring paragraph and shifting
`nemo_testcase_recipe.py` (two pure-insertion hunks, +7 and +6 lines, for
the two cards' explicit config choices). Each round shifted lines below its
own edit point(s) by a RIGID delta, computed fresh from `git diff
--unified=0` and verified against the actual pre/post line count of every
hunk — never assumed constant across rounds. Every `CITATION_MAP` entry
keyed on either file needed the shift (same delta on both endpoints, same
pinned extent, same anchor text where the text itself did not change) —
applied by a small script each time, never by hand, and checked by
re-running `audit_map()` (zero failing rows over the FULL map, not just the
touched entries) before committing after EVERY edit. One entry's cited TEXT
itself changed each round, because it anchors on
`nemo_stage_momentum_wzv_executes`'s own final statement: `getattr(...,
False))` → `..., True))`, then → `return bool(getattr(config, "eos", ...))`,
then (final shape) → `return bool(config_split)` at its shipped range
`1668-1705`. Nine citations into `ocean_model_latlon_cgrid.py` plus one into
`nemo_testcase_recipe.py` inside the round-8 master receipt (the citation
gate's `DEFAULT_RECEIPT`) needed the identical shift in prose each round;
fixed the same way, verified the same way, every time.

## OPEN — Round 164

1. **Walk the stage-2 entry velocity.** Round 162 retracted the campaign's
   own prior summary: round 159's evidence JSON (not its prose) shows the
   oracle's stage-2 ENTRY velocity — the stage-1 output velocity handed to
   the momentum continuity solve at
   `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360` — removes 55.06% of
   the remaining `2.334682e-13` m/s residual by rms and 67.33% by max, not
   the ~1.5e-06 (transport-form) number round 161's OPEN misquoted. Run the
   VELOCITY-FORM null control FIRST (round 159's null ran only in the
   transport-form arm). Before substituting any operand, rank every arm in
   round 159's own JSON rather than reading its OPEN section, and tag the
   substituted continuity call instead of counting it (round 162 finding E).
2. **Vertical diffusion stays the day-240 process owner to walk in the
   developed state.** This round's `--decompose 240` is the SPATIAL pattern
   (region/depth/field), unchanged by the landing; it is not evidence about
   which PROCESS carries day 240 in the now-landed arm. Round 162 measured,
   at the one-step developed state, that vertical diffusion is still the
   largest row and that it and advection do NOT cancel (correlation
   +0.0027). Re-run round 152/161's one-step process ranking in the NOW-LANDED
   arm (it was previously run in the private-arm candidate, which is the
   same code, so this is expected to reproduce round 161/162's numbers, and
   that reproduction is itself worth stating rather than assuming).
3. **The reviewer's carried caveat from round 160** (compiled-scheduling
   floor bound, 1.0 of 2 units in the last place on one arm) is still
   unreproduced on an unrelated arm; still open, still registered, not
   re-measured this round.
4. **ORCA2-zps: measure it, then answer the DECISION_NEEDED above.** This
   round's review upgraded ORCA2 from "unmeasured-with-spec" (round 160's
   disclosure) to "resolves the exact same program, actively excluded by its
   own explicit config choice (`nemo_stage_momentum_wzv_split=False`) until
   measured" (round 163's fix). The next ORCA2 round should run this route's
   certified ladder against ORCA2's own oracle record (same instrument this
   round used for GYRE, `--trajectory-only` comparison) and report whether it
   helps, hurts, or is inert there — then the operator decides whether to
   flip ORCA2's own config field to `True`. Do not flip it without that
   measurement; the explicit `False` on ORCA2's own card config is what
   prevents that from happening by accident, and any card that later forgets
   to set the field at all is caught by a raise, not a guess.
5. **Add ORCA2-zps as its own row in `nemo_testcase_l2_gyre_decision43_gate.py`'s
   `_card_execution`.** It has never been in that census (round 160's gap,
   not introduced by round 163), and this round's fix worked around that by
   checking ORCA2's resolved config directly in a test rather than through
   the gate. Building the row needs `build_orca2_zps_card`'s deck files,
   which this round did not touch — noted as a completeness gap, not
   re-attempted here.
