# NEMO testcase L2 GYRE Round 163 receipt — the second continuity solve LANDS

Date: 2026-09-24

Status: **LANDED** — round 160's second per-stage continuity solve is now the
production default for the GYRE card, under Decision 55 (note AT, the amended
year gate).

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

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round163.md`, committed as
`987af9a71` before any round-163 measurement ran. The flip is
`af15e39c4`. The before arm is round 160's own already-measured base commit
`de4e0cfe35daa6e6e5282a9b192a78abffb47ada` (its ladder, month and year members
are reused unchanged, since rounds 161 and 162 both touched zero files under
`packages/` — checked again this round: `git diff --stat de4e0cfe35da..b18cfc276e39 -- packages/` shows only round 160's own private-arm commits, round 161's and
round 162's measurement/test-only commits). The candidate arm is measured at
`8ec2fba50296ffe51dab7c0c960ff8c9c5b6f4f1` (the citation-map re-anchor commit;
no production line changed between the flip and this commit). Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round163/`.

## The flip

Decision 55 (note AT): a NEMO statement that is (1) cited from compiled
`ppsrc`, (2) proven one-variable, and (3) takes a certified trajectory row
from DEBT to AT-BAR, LANDS even when day-240/360 T rms move by less than 1e-3
relative. Round 160 already proved all three; round 163 flips the production
default.

`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py`:
`_NEMOWSRK3TestHooks.nemo_stage_momentum_wzv_split` default `False` → `True`,
and `nemo_stage_momentum_wzv_executes(config, hooks=None)`'s no-hooks
`getattr` fallback `False` → `True` to match, so the model and the
Decision-43 card census agree with no hooks passed at all — which is every
production card. Grepped: the only `_NEMOWSRK3TestHooks(` call anywhere in
`packages/` is the model's own `_nemo_ws_test_hooks or _NEMOWSRK3TestHooks()`
default, so this two-line flip is the whole production change. The arm stays,
not deleted: an explicit `nemo_stage_momentum_wzv_split=False` remains the
one-variable way to reach the pre-round-160 single shared solve, and three of
this round's own structural tests (in the round-160 test file, updated this
round) use exactly that opt-out to prove it still works.

### The compiled statement, re-cited

`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/stprk3_stg.f90:360` (the raw stage
velocity, velocity indicator) and
`GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/divhor.f90:126-130` (its continuity
solve) for momentum; `GYRE_OMIP_L2_P3_SM_R156ST2/BLD/ppsrc/nemo/traadv.f90:274`
for the tracer re-solve, unchanged. Same citations round 160 gave; re-quoted
here rather than only referenced, per note AT item 1.

### Card census, re-verified from each card's resolved configuration

| card | momentum integrator | momentum advection | continuity solve | runs the candidate | runs at this tip |
|---|---|---|---|---|---|
| GYRE-zco | rk3_ws | vector_invariant | nemo_literal | **yes** | **yes** |
| LOCK_EXCHANGE-zco | rk3_ws | flux_form | generic | no | no |
| OVERFLOW-zps | rk3_ws | flux_form | generic | no | no |
| NEMO-GYRE recipe | rk3_ws | vector_invariant | generic | no | no |
| DINO nemo_dino_kamm | euler | vector_invariant | nemo_literal | no | no |
| DINO nemo_dino_kamm_mlf | euler | vector_invariant | nemo_literal | no | no |

Re-run this round via
`tests/ocean/fidelity/test_nemo_testcase_l2_gyre_round160_wzv_split.py::test_the_admission_gate_census_uses_the_model_s_own_predicate`
(updated: `executes_at_this_tip` now asserts `["GYRE-zco"]`, not `[]`) and via
each card's own gate battery below. GYRE-zco is the only card whose resolved
configuration takes the RK3-WS vector-invariant literal-continuity deck
together; every other card is out on the momentum integrator, the momentum
advection form, or the continuity-solve selector, exactly as round 160 found.

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
sonnet) reviewed the round's diff — the two model-file edits, the
census-script comment update, the round-160 test file's three updated
assertions, and this receipt — against the compiled source and the
Decision-55 rule. Verdict: **SHIP**. Findings and disposition:

1. The `getattr(hooks, "nemo_stage_momentum_wzv_split", True)` fallback
   changes behavior for any FUTURE caller that passes a `hooks` object
   without setting the field (e.g. a test hooks instance built via
   `_NEMOWSRK3TestHooks(some_other_field=...)`), not only the no-hooks
   production path. CLOSED — checked: `_NEMOWSRK3TestHooks` is a
   `NamedTuple`, so any partially-specified construction still carries the
   field's own default (`True`), which is the same value the `getattr`
   fallback now provides; the two can never disagree. Grepped every
   `_NEMOWSRK3TestHooks(` construction: none sets `nemo_stage_momentum_wzv_split`
   to `False` outside the three tests written to do so on purpose.
2. The receipt's citation re-anchor touches `scripts/validate/...` and
   `docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_phase3_round8_receipt.md`
   — a doc outside this round's stated file list. REGISTERED, not a defect:
   the round-8 receipt is the citation gate's `DEFAULT_RECEIPT`, so leaving
   its nine stale citations unrepaired would fail the push gate's own
   citation test; the fix is mechanical (a same-line text substitution, +9 on
   each of 9 already-identified stale citations) and was verified against
   `audit_map()` returning zero failures both before and after.
3. No finding on the physics: the reviewer opened
   `stprk3_stg.f90:360`/`divhor.f90:126-130`/`traadv.f90:274` directly and
   confirmed the receipt's citations still say what the receipt says they
   say; confirmed the arm-off path (`nemo_stage_momentum_wzv_split=False`)
   is unreachable from any card's resolved configuration going through
   `nemo_stage_momentum_wzv_resolved`, so no other certified card can
   silently pick up the split through the default change.

## Push gate and this round's own tests

Citation gate plus the five push-gate files plus this round's own updated
structural test file, on the committed tree:

> 162 passed in 929.79s (0:15:29)

Citation gate detail: `audit_map()` over the FULL map (not just this file's
39 re-anchored entries) returns zero failing rows; `run(DEFAULT_RECEIPT,
DEFAULT_HEADING)` returns `status: PASS`, zero unmapped citations, zero
failures, zero map-audit failures, all self-tests fired.

## What changed in the citation map, mechanically

Three edits to `ocean_model_latlon_cgrid.py` (a comment growing 7→12 lines
at the field default, a docstring holding steady at 6 lines, and a second
docstring growing 5→8 lines) shift every line at or after 1296 by +5, and
every line at or after 1666 by +9 (delta computed from `git diff
--unified=0`, verified against the actual pre/post line counts of each
hunk). 39 `CITATION_MAP` entries keyed on this file needed a RIGID shift
(same delta on both endpoints, same pinned extent, same anchor text) —
applied by a small script, not by hand, and checked by re-running
`audit_map()` before committing. One entry's cited TEXT itself changed
(`nemo_stage_momentum_wzv_split", False))` → `..., True))`); it is
re-anchored by hand to its new range `1660-1674` and its new text. Nine
citations into this same file inside the round-8 master receipt (the
citation gate's `DEFAULT_RECEIPT`) needed the identical +9 shift in prose;
fixed the same way, verified the same way.

## OPEN — Round 164

1. **Walk the stage-2 entry velocity.** Round 162 retracted the campaign's
   own prior summary: round 159's evidence JSON (not its prose) shows the
   oracle's stage-2 ENTRY velocity — the stage-1 output velocity handed to
   the momentum continuity solve at `stprk3_stg.f90:360` — removes 55.06% of
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
4. **ORCA2 stays UNMEASURED-WITH-SPEC** for this route (round 160's
   disclosure, unchanged).
