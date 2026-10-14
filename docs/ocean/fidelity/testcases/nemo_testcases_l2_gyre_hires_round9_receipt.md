# NEMO testcase L2 GYRE high resolution — round 9 receipt

Date: 2026-10-14

Incoming lane tip: `387e7a14a3e8fb4b826db868f991c5339b6d3836`

Work copy: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/claude-gyrehr-r9`
(uncommitted; this round was written by Claude, not codex).

Evidence: `/home/dbalwada/.claude/jobs/b3690d5d/tmp/claude_gyre_r9/impl/evidence/`
(to be copied by the operator to `gyrehr_rounds/round9/`).  Every gate below
ran on CPU under the certified fp64 scalar-libm policy against the admitted
scalar-math records.  The tree was dirty (`LEGOESM_GATE_ALLOW_DIRTY=1`); each
JSON records the diff hash, and `final_code_sha256.txt` pins the four source
files that produced the final numbers.

## Verdict

**HELD at a measured owner whose fix is an already-held candidate.**

1. **F5 — the remaining step-2 temperature owner is measured, not inferred.**
   NEMO's executed `avt` substituted at the vertical-solve seam takes the
   step-2 solved temperature from `1.81572090696136e-09 K` to
   `1.177724584522366e-12 K`, the pre-solve floor.  The `avt` difference is
   99 % one thing: the Richardson-number Prandtl factor in the 227
   first-interface cells where it is not clamped (NEMO's `avt` in those 227
   cells alone gives `1.9632295789051568e-11 K`).  Eager evaluation shows the
   same difference, so it is not JIT rounding.  Routing the step-entry sea
   surface into the TKE shear divisor — round 166's held one-degree
   candidate — removes it: `avt` falls to ULP level and the step-2 solved
   temperature to the pre-solve floor (numbers in F5).  This is a
   transcription (time-level) difference, not composition.
2. **The fix is not landed.**  Round 166's manifest holds it until a new gate
   changes one of two facts: the one-degree day-240 year veto (Decisions
   45/55) and ORCA2-zps, which executes the same route unmeasured.  This
   round measured neither, and this round's own rule (keep every card that
   runs a shared change at or closer to NEMO) cannot be met for ORCA2.
   DECISION_NEEDED below.
3. **F3 fixed.**  The stage-3 two-band solar stretch now uses the stage
   weights' own `1+r3t(Kmm)`; with NEMO's sea surface it matches NEMO's
   record in 9,600/9,600 columns and NEMO's `tra_qsr` accumulator in
   288,000/288,000 words (was 4,252 columns and 39,762 words off).
4. **F8 guarded, F1/F2 documented, one-degree distance guarded** (sections
   below).

## Did round 8's solar fix help at quarter degree?

Yes, by a factor of about 2 at step 2 and 3 by step 10.  Max |legoESM − NEMO|
at the scalar-math entry records; round 7 is the commit before the fix
(packages/src identical to `63cebeb37`), round 8 is `387e7a14a`:

| kt | r4 T r7 | r4 T r8 | r4 SSH r7 | r4 SSH r8 | r4_smag T r7 | r4_smag T r8 |
|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3.339e-09 | 1.816e-09 | 1.848e-11 | 1.848e-11 | 3.340e-09 | 1.816e-09 |
| 5 | 1.093e-08 | 2.652e-09 | 6.192e-11 | 2.455e-11 | 1.089e-08 | 2.789e-09 |
| 10 | 2.206e-08 | 8.384e-09 | 1.926e-10 | 3.552e-11 | 2.272e-08 | 7.634e-09 |

Not every row improved: r4_smag kt10 U went from `9.729e-10` to
`1.319e-09`.  Sources: `gyrehr_rounds/round7/r4{,_smag}_scalar_ladder.json`
and `gyrehr_rounds/round8/r4{,_smag}_scalar_ladder.json`.

## F5 — owner of the step-2 temperature difference

**Finding (review F5).**  Round 8 named TKE-derived `avt` as the next owner
because it is the first row of a fixed list with any unequal word.  No
substitution was run, so that was an inference.

**Measurement.**  New committed mode `--step2-vertical-owner-only` of
`nemo_testcase_l2_gyre_hires_ladder_gate.py` (`run_step2_vertical_owner`),
same bridge as round 8: NEMO's kt=2 primary state, legoESM's carried closure
memories.  Each arm changes one input of the production-jitted step and is
scored against NEMO's executed `tra_zdf` solution (`sol_T_pre_clamp`, all
288,000 wet cells).  Final-code run: `step2_vertical_owner_final.json`.

| arm (one changed input) | solved-T max abs (K) | max ULP | unequal |
|---|---:|---:|---:|
| production step | `1.81572090696136e-09` | 511,080 | 260,529 |
| seam present, nothing replaced | `1.81572090696136e-09` | 511,080 | 259,414 |
| NEMO `avt`, all interfaces | `1.177724584522366e-12` | 1,014 | 255,433 |
| NEMO `avt`, 227 Prandtl cells only | `1.9632295789051568e-11` | 5,526 | 259,396 |
| NEMO `avt`, every other interface | `1.81572090696136e-09` | 511,080 | 255,451 |
| NEMO kt=1 `avm` as carried `avm_k` | `1.81572090696136e-09` | 511,080 | 260,529 |
| NEMO kt=1 `avt` as carried `avt_k` | `1.81572090696136e-09` | 511,080 | 260,529 |
| same step evaluated eagerly (no JIT) | `1.8157244596750388e-09` | 511,081 | 256,320 |
| round-166 shear routing split | `1.1795009413617663e-12` | 1,015 | 256,573 |

Controls, all measured in the same run: the seam is not bit-passive (64,734
solved words move by at most 3 ULP, `1.0658141036401503e-14 K`, against the
plain step) but keeps the plain maximum; scaling NEMO's `avt` by `1+2^-40`
moves 65,818 solved words, so the seam can see one input; NEMO's `tra_zdf`
`avt` equals its `zdf_phy`-exit `avt` in all 278,400 interior words.

**Decomposition.**  Above its floors NEMO writes `avm = zav` and
`avt = pdlr*zav`, so `avt/avm` isolates the inverse Prandtl number and a
residual common to both cancels in it.  At kt=2 legoESM's `avt` differs in
77,796 words (`5.357229237046984e-10 m2/s` at `[j=27,i=120]`, first
interior interface, NEMO `jk=2`).  Two
parts: (a) the Prandtl ratio differs in exactly the 227 first-interface
cells where NEMO's `pdlr` is not clamped (`0.1 < pdlr < 1`; the other 9,373
first-interface cells sit at 0.1 in both models), relative `4.1e-10` to
`3.28e-7`, spread over the basin with 7 on the wet boundary rows; (b) a factor
common to `avt` and `avm` differs in 13,397 cells on the first two interfaces,
relative at most `7.07e-9`.  Part (a) owns 99 % of the solved residual
(table).

**Owner.**  `zri = rn2b*p_avm/(p_sh2+rn_bshear)` feeds `pdlr`.  The carried
`p_avm` is ruled out by its memory arm.  Eager evaluation gives the same 227
cells to the last digit shown (eager and JIT `avt` differ by at most 44 ULP),
so this is not the JIT rounding of `rn2` that rounds 106-108 found at one
degree.  The shear `p_sh2` divides by `e3w*(1+r3u(Kmm))*e3w*(1+r3u(Kbb))`,
and NEMO's closure call binds both levels to the step-entry slot.  legoESM
hands its shear the stage-3 half-step sea surface, the slot its implicit
tracer solve needs.  Round 166 found this at one degree on a developed state
and made `p_sh2` bit-exact with a one-variable routing split.  Re-applied
here as the single changed variable (diff: `r166split_on_round9.diff`), the
split removes both parts: `avt` differs by at most 42 ULP
(`9.107298248878237e-18`), `avm` by at most 22 ULP, zero Prandtl or
common-factor cells remain, and the solved temperature falls to the
pre-solve floor (`step2_vertical_owner_r166split_final.json`).  CONFIRMED.

Why it starts at step 2: from rest the shear is zero, so the time level of
its divisor cannot matter at kt=1; from step 2 both the shear and the
half-step sea-surface change are live.  It is not quarter-degree specific.

**Disposition.**  A transcription (time-level routing) difference, not a
composition one.  **Not landed**, see DECISION_NEEDED: the held round-166
manifest
(`manifests/nemo_testcase_l2_gyre_round166_held_shear_routing_split.patch`)
says not to land it until a new gate changes its two facts — the one-degree
day-240 year veto under Decisions 45/55, and ORCA2-zps, which executes the
same route, unmeasured.  This round's own rule for shared changes cannot be
met for ORCA2 either.  What landing would do on the GYRE family is measured
below (ladders with the split).

**Reversion.**  Not applicable (nothing landed).  The instrument fails closed:
plant and seam rows are in every report, and its verdict names
`NO_AVT_RESIDUAL` when a tree has no `avt` owner left (the split arm).

## F3 — the live-stretch composition at the solar boundary

**Finding.**  legoESM's stage-3 two-band solar stretch was
`0.5*(h_b+h_a)[top]/h_ref[top]`; NEMO's is `1+r3t(Kmm)` with
`r3t(Kmm) = r1_2*(r3t(Kbb)+r3ta)` written at stage 2 and `r3t = ssh*r1_ht_0`.

**Measurement** (`--stage3-kmm-stretch-only`, `run_stage3_kmm_stretch`;
NEMO's own kt=2 entry and stage-3 sea surfaces as inputs, so every unequal
word is composition):

| stretch composition | `1+r3t(Kmm)` words unequal / 9,600 | `tra_qsr` accumulator words unequal / 288,000 |
|---|---:|---:|
| round-8 thickness ratio | 4,252 (max `2.22e-16`, 2 ULP) | 39,762 (max `3.388e-21 K/s`, 61,440 ULP) |
| NEMO statement, mean of ratios | 0 | 0 |
| legoESM stage weights `q_Kmm` (ratio of mean SSH) | 0 | 0 |

`r3t(Kbb)` and `r3t(Kaa)` rebuilt as `ssh*r1_ht_0` match NEMO's words 0/9,600,
which validates the reciprocal.  This 4,252-column difference is the owner of
the 17-ULP debt round 8 reported at that boundary.

**Disposition — fixed.**  The two-band branch now evaluates the stage-3 field
on the stage weights' own `q_Kmm` (zero on dry columns, as before), a
two-line change in `_step_impl`.  The stage weights' composition (ratio of
the mean sea surface) is real-equivalent to NEMO's mean of ratios and matched
it on all 9,600 columns here; it is not proven equal in general (the stage-1
helper already writes NEMO's form; a stage-2 twin is the pattern if a later
step shows a difference).  The chlorophyll (RGB) branch has the same
thickness-ratio form; ORCA2 runs it and it is not touched here.

**Effect.**  Ten-step ladders below.  At step 2 the solved-T maximum is
unchanged to every printed digit (the `avt` owner dominates).

**Reversion.**  `test_stage3_two_band_solar_uses_the_stage_weights_kmm_stretch`
(production step, one-degree card, live sea surface) PENDING_F3_REVERSION.

## F8 — two-band solar scheme on an integrator it is not wired for

**Finding.**  Only the rk3_ws stage program replaces the shared pipeline's
step-entry qsr_2BD field with NEMO's stage-3 Kmm call.  Any other integrator
would keep the step-entry field as the only solar source.

**Disposition — guarded.**  `_validate_config` raises when
`shortwave_penetration.scheme == "nemo_qsr_2bd"` and the tracer integrator is
not `rk3_ws`.  The misleading `combined.py` comment that called the pipeline
field "the WS-RK3 Kbb evaluation" now says NEMO has no such evaluation.

**Evidence and reversion.**
`test_nemo_qsr_2bd_refuses_integrators_without_its_stage3_kmm_call` passes
(rk3 alone and rk3_ws with the scheme both construct; rk3 with the scheme
raises).  On a scratch copy with only the guard removed it fails with
`DID NOT RAISE` (`f8_test_on_reverted_copy.log`).

## F1 — the one-degree card against NEMO, before and after round 8

The round-8 receipt reported only movement relative to round 7.  Against
NEMO's scalar-math records, from the lane's own artifacts
(`gyrehr_rounds/round7/shipped_trajectory.json` before,
`gyrehr_rounds/round8/shipped_trajectory.json` after; the review reproduced
both bit for bit), the ten-step gate moved **37 rows closer, 15 farther, 18
unchanged**:

| row | before (round 7) | after (round 8) |
|---|---:|---:|
| kt3 T | `1.376582e-08` | `1.130699e-08` |
| kt10 T | `1.249712e-07` | `8.730964e-09` |
| kt10 S | `3.044999e-08` | `4.556135e-09` |
| kt10 U | `1.786945e-09` | `2.553390e-10` |
| kt10 V | `2.373224e-09` | `8.790116e-10` |
| kt10 SSH | `1.274973e-09` | `5.234391e-10` |

The 15 farther rows are the 14 barotropic `uu_b`/`vv_b` rows at kt4-kt10
(+0.4 % to +4.5 %) and kt5 U (`3.749109e-10` to `4.966084e-10`).  The card is
not bit-exact past kt=1 under the zero-word bar (kt2 T has 6,832 unequal
words, labelled AT-BAR only by the relative 1e-15 bar); the fix lost nothing
that was exact.  Longer gates, measured by the review at the same two
commits and not re-run here (`review_gyre_r8/find/`): 60-step T RMS
`4.871e-07` to `4.782e-09` K; from-rest year T RMS at day 30
`2.343e-06` to `4.021e-08`, day 240 `6.582e-05` to `7.515e-07`, day 360
`5.408e-05` to `1.243e-06` K, with the velocity bursts gone.

**Why the regression gate failed.**  Not because the trajectory moved.  The
oracle-relative gate fails when any single cell gets worse against NEMO by
more than 2 row-scale ULP; 52 of 70 rows had at least one such cell even where
most cells improved (kt10 T: 13,200 cells improved, 4,403 worsened, 3,125
beyond the bar).  Round 8's sentence is revised below.

## Retractions and revisions (Rule 11)

1. **Round 190 (one-degree lane, 2026-09-27, `bd225cab0`) — RETRACTED in
   part.**  What it was: a measured difference between the observer's Kbb
   QSR field and the production Kbb QSR field in 10,200/18,000 wet cells
   (`1.713729456486868e-10 K/s`, carrying `2.467770444880557e-06 K` per step
   by day 180), classified as "an observer classification artifact, not a
   model shortwave statement", with the instruction to correct the observer,
   not model physics.  What killed it: the production step subtracted a
   live-stretch Kbb field at stage 3 while the shared pipeline had deposited a
   static-stretch one, so that difference stayed in the tracer RHS as a false
   solar source; NEMO has no Kbb solar evaluation at all.  Round 8 removed it;
   the round-8 review reproduced the 10,200-cell count on a synthetic sea
   surface before the fix (0 after) and measured the one-degree year gap 43-365
   times smaller after it (F1).  Consequence: round 186's ranking of
   shortwave as the largest pre-day-180 carrier, which round 190 declared not
   usable, is reinstated as PLAUSIBLE only; it was not re-measured at round
   186's commit, whose trajectory differs from today's.  A pointer to this
   section now heads the round-190 receipt.
2. **Round 8, verdict paragraph — REVISED.**  "The acquired executed
   vertical-solve record localizes its first input boundary to TKE-derived
   `heat_K` (`avt`)" was list order, not localization.  Replaced by F5's
   substitution arms: `avt` does own the step-2 residual, and within it the
   Prandtl factor of 227 cells, set by the shear divisor's sea-surface time
   level.
3. **Round 8, one-degree section — REVISED.**  "The trajectory regression
   correctly fails because fixing that shared kernel moves the trajectory" is
   replaced by F1's reason (cell-level worsening beyond 2 ULP in 52 rows while
   most cells improved).
4. **Round 8, OPEN item 2 — RETRACTED.**  "acquire only the TKE operands
   needed to split input-history from the `tke_avn` coefficient statement" is
   not needed: no new NEMO record was required to name the owner.

## Ten-step scalar ladders after round 9

PENDING_LADDERS

## What landing the routing split would do (scratch arm, not landed)

PENDING_SPLIT

## Shipped one-degree card after round 9, and its new distance guard

PENDING_ONEDEG

## Long-run disposition

PENDING_LONGRUN

## DECISION_NEEDED

PENDING_DECISION

## Inherited pins / OPEN

Inherited unchanged: every deck value, the shipped one-degree deck and its
digest, the `r4` and `r4_smag` namelists, 31 levels, RK3, split-explicit free
surface, Redi tracers, TKE/EVD selection, the 360-day forcing clock, the
scalar-libm CPU certification policy, the zero-word ladder bar, daily-mean
conventions and the explicit physical-GPU requirement.  No configuration
knob, default or selector was added or changed; the F8 guard only refuses a
combination no card uses.

OPEN, in order:

1. The decision above (routing split).
2. After it: the r4/r4_smag ladders' first remaining owner (the split arm's
   kt=3 rows say where).
3. The chlorophyll (RGB) stage-3 branch has the same thickness-ratio stretch
   that F3 removed from the two-band branch; ORCA2 runs it (its lane).
4. The disabled seam of the vertical-solve test input is not bit-passive (3
   ULP); it keeps the maximum, which is all F5 uses.

## Tests and evidence

PENDING_TESTS

## Compiled-source citations

The step program evaluates the closure once, before the stages, with both
formal time levels bound to the step-entry slot, at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3.f90:168-216`.
The shear production divides by the Kmm and Kbb face metrics, both live
`(1+r3u)` factors, at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdfsh2.f90:99-103`.
The Richardson number and the inverse Prandtl number are formed at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90:389-404`
and applied to `avt` (not `avm`) at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/zdftke.f90:681-692`.
Stage 2 writes the stage-3 `r3t(Kmm)` as the mean of two ratios at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:221`.
Stage 3 calls `tra_qsr` once, with `Kmm`, inside the tracer call order at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/stprk3_stg.f90:691-725`,
and the two-band kernel uses `1+r3t(Kmm)` in both depth and thickness at
`GYRE_OMIP_L2_P3_SMW/BLD/ppsrc/nemo/traqsr.f90:615-645`.
