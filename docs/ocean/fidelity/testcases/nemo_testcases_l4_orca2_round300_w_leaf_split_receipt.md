# ORCA2 round 300 — kt=7 W leaf split

Date: 2026-10-16

Status: **HELD**.  On both **independent OMT-4** and **given NEMO's
entry**, the first source-ordered non-bit W-builder boundary is the U-transport
difference in NEMO's transport-form horizontal divergence.  It differs on
429,642/430,552 active values, with maximum absolute difference
147,510,270.990179 at `[j=147,i=48,k=25]` and RMS
1,079,264.0190773946.  No production package, card, selector, carried state,
stabiliser, sea-ice field, or `unmeasured_features` entry changes in this
round.  The complete fold unit remains private and atomic.

**Main finding (review response, below):** the card's stage-1 U transport is
the same quantity as NEMO's and is aligned cell for cell, but in the fold band
it is wrong by orders of magnitude (card `7.726e7` vs NEMO `1.346e5 m3/s` at
`[147,47,25]`; band max difference `7.91e7` vs NEMO band max `1.07e6`), which
carries about 98% of the squared domain difference; the interior differs by
about 14% RMS.  The earlier "independent ratio/clock pair" row compared two
different time levels and is corrected below.

## Provenance and admission

- Lane base: `deee20d23`; frozen preregistration: `31290fd4d`.
- Measurement tree: `ac82e1616ab6b38ddce19f63e8420a99425a5cae`, clean,
  CPU, fp64, production JIT, libm.
- Record:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round299/acquisition/orca2_omt4_stage7_w_10step_np2`.
- Admission: `PASS_R299_OMT4_KT7_W_ADMISSION`; both self-describing rank
  files cover the 148x180 domain exactly once, and inherited stage frames and
  kt=10 restarts are byte-identical to the admitted round-297 baseline.
- Raw measurement SHA-256:
  `cbab49a6d8ccb3855915a4d4c97864e3887150db7e4afd36a54a31ce13db3ca7`.
- Classification SHA-256:
  `daeff4b8cb779848b4293754b35dddc60f37d4b2a4727a8b80ed9b379b853014`.

Both passive traces are array-identical to their ordinary completed states
for every carried field.  NEMO's replay is exact for `ww` and its area-scaled
transport.  The candidate `ww` replay is within `6.938893903907228e-18 m/s`;
the final transport replay, scored in W-equivalent units so ORCA2's
`O(1e10 m2)` cell area cannot amplify a sub-floor W difference, is within
`1.3877787807814457e-17 m/s`.  The raw area-product row is nevertheless
reported below without normalization.

### Instrument corrections retained

The frozen preregistration was not edited.  Three fail-closed corrections
preceded the admitted measurement: the resolved T mask replaced a nonexistent
diagnostic attribute; the 31-interface NEMO W record was aligned with the
candidate's 30 active layers; and NEMO's stage-1 `rn_Dt/3` ratio/clock pair
was kept distinct from the candidate's production full-step pair.  A later
lookup treated the pseudo-operand `w_dt` as a record field and failed before
classification.  Finally, the unnormalised transport replay refused because
a `6.94e-18 m/s` W residual became `5.96e-8 m3/s` after area multiplication;
the representation-safe replay correction above was committed before the
successful measurement.  None of these refused attempts produced a physics
claim.

## Compiled source and executed order

The cited record is the compiled `ORCA2_OMIP_L4_R299W7` configuration.
Enclosing subroutines (pinned by the citation gate's routine check):
`tra_adv_trp_t` for both `traadv.f90` ranges, `div_hor_RK3_t` for
`divhor.f90`, `wzv_RK3_t` for `sshwzv.f90`, and `stp_RK3_stg` for
`stprk3_stg.f90`.  The `traadv.f90` W range in this record build contains
the acquisition's write-only record block around the `wzv` call; the model
statements in it are the `wzv` call and the final area product.  It
zeros the bottom U/V/W transports at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/traadv.f90:248-252`, executes the W
diagnosis and final `e1e2t*ww` product at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/traadv.f90:267-315`, and evaluates the
transport-form U difference before the V difference at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/divhor.f90:132-139`.

The QCO recurrence sets bottom W to zero and integrates upward at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/sshwzv.f90:271-298`.  Its stage-1
clock is `rn_Dt/3`, followed by the reciprocal, at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:136-137`.  These are
live branches under the admitted OMT-4 namelist, not dead-arm citations.

## Source-ordered measurement

Both labels give the identical table.  The floor is `2e-10` in the native
units of each row.

| Statement | Unequal / support | Max abs | RMS | Argmax `[j,i,k]` |
|---|---:|---:|---:|---|
| bottom zero | 0 / 16,433 | 0 | 0 | `[1,49]` |
| U difference | 429,642 / 430,552 | 1.47510270990179e8 | 1.0792640190773946e6 | `[147,48,25]` |
| V difference | 429,735 / 430,552 | 1.7358593746527478e8 | 1.0320759593368458e6 | `[147,132,27]` |
| horizontal sum | 430,552 / 430,552 | 2.0169372251798508e8 | 8.622053691944386e5 | `[147,48,25]` |
| transport divergence | 430,552 / 430,552 | 2.6023067008330836e-2 | 1.0794110578633983e-4 | `[147,48,25]` |
| live thickness | 430,552 / 430,552 | 6.668582857059846e-2 | 1.1693549040022798e-3 | `[147,50,23]` |
| flux divergence | 430,552 / 430,552 | 2.6023067008330836e-2 | 1.0794110578633983e-4 | `[147,48,25]` |
| Kaa-Kbb ratio delta, time-matched (corrected) | 16,433 / 16,433 | 9.4070653952792217e-3 | 2.1140007761263205e-4 | `[141,125]` |
| stretch rate | 430,552 / 430,552 | 8.836174225055932e-5 | 8.009334080397122e-7 | `[147,48,27]` |
| W recurrence | 430,529 / 430,552 | 1.9152110475369796e-1 | 1.5744398704267452e-3 | `[147,48,0]` |
| area product | 430,529 / 430,552 | 1.4843986124373436e9 | 1.222542183061254e7 | `[147,48,0]` |

Replacing candidate U transport alone leaves W maximum error
`0.19217452883025382 m/s`; replacing U+V leaves
`0.0007441398648953929 m/s`.  Area, live Kmm thickness, and mask do not
close that residual.  The replay becomes exact only when the recorded
Kbb/Kaa ratios and stage clock are added together.  The preregistered claim
that the complete divergence alone closes W is **REFUTED**, but (corrected
after review) the remaining member is the step's sea-level change, not the
clock: see "Correction 2" below.  The ratio row in this table was rebuilt at
matched time level (r300d re-measurement); the originally committed row
(`5.0750835119969487e-2` max, `1.3644301289032852e-3` RMS) was mis-built.

### Fold band versus interior (added at completion, same raw record)

The table above is domain-wide.  Split into the three fold rows and the
interior (the round-298 `_regions` split), the debt is concentrated in the
fold band:

| Row | Fold band unequal / max | Interior unequal / max | Fold / interior max |
|---|---|---|---:|
| U difference | 4,149/4,164, 1.47510270990179e8 | 425,493/426,388, 2.569318308591891e7 | 5.7 |
| V difference | 4,145/4,164, 1.7358593746527478e8 | 425,590/426,388, 6.600581266266088e7 | 2.6 |
| W recurrence | 4,164/4,164, 1.9152110475369796e-1 | 426,365/426,388, 2.6643902175528533e-3 | 71.9 |
| U+V substituted, W | 4,164/4,164, 7.441398648953929e-4 | 426,388/426,388, 1.4412554193721734e-4 | 5.2 |

Every domain-wide maximum sits in the fold band.  The interior is non-bit
too, as expected after six steps of independent trajectory, so "first in
source order" names the first DIFFERING OPERAND, not an origin.

### What the W-leaf finding is, and is not (added at completion)

- **The W builder's own statements are exact on this record.**  From NEMO's
  recorded operands, the shared production helpers reproduce NEMO's `ww` and
  area-scaled `trp_w` with zero difference (both labels).  With every
  candidate operand replaced by NEMO's, the candidate program's W is exact
  everywhere.  So `div_hor_RK3_t`'s `np_transport` divergence, the
  `wzv_RK3_t` QCO recurrence and the `tra_adv_trp_t` area product carry no
  statement defect: there is **no W statement to land** under D96 or D122.
- **The fold-band W debt is carried by operands produced upstream:** the
  stage-1 U and V transports jointly (neither alone), plus the step's
  sea-level change entering the stretch term (corrected; the stage clock is
  a time-level representation, not a member).  The U difference is the first of these
  in compiled order; it is held with the numbers above.
- **The two labels' tables are array-identical** (every row, arm and replay).
  The card's kt=1 entry equals NEMO's admitted kt=1 entry (phase-2y
  admission; PLAUSIBLE as the reason, not re-measured here), so the second
  label confirms the program but is not independent evidence.

## Prediction ledger

| ID | Verdict | Result |
|---|---|---|
| R300-P1 | **CONFIRMED** | Admission, provenance, both labels and all controls pass. |
| R300-P2 | **CONFIRMED** | Candidate and oracle replays pass in W-equivalent units; the raw area-product row remains reported. |
| R300-P3 | **CONFIRMED** | U difference is the first non-bit boundary on both labels. |
| R300-P4 | **REFUTED** (reason corrected) | Divergence substitution alone leaves `7.44e-4 m/s`; the remaining member is the step's sea-level change (time-matched ratio residual max `9.41e-3`), not the stage clock. The original "ratio/clock pair independently required" wording rested on a row that compared different time levels. |
| R300-P5 | **CONFIRMED** | Both labels independently produce the same source order and magnitudes. |
| R300-P6 | **CONFIRMED_MEASUREMENT_ONLY** | No production or private constituent lands. |

## Controls, validation, and review

All eight classifier plants refuse with exit 2 and distinct
`STATUS PLANT-FIRED` reasons: admission, label coverage, source order,
passivity, candidate replay, oracle replay, first-leaf selection, and the
normal one-bit control.  Plant log SHA-256:
`62b18f4b9fe00c9bddd875519ea0f7d94fe9b9874999fb946cbb00d6ae6c7fe7`.
Decision 127 is exercised separately: zero versus a NEMO nonzero subnormal is
counted as one raw difference, one subnormal match, and zero scored debt;
the normal one-bit control remains unequal.

Citation-gate, focused-test, full fidelity-battery, and independent-review
results are recorded below after execution.

### Completion record (Claude, round 300 continuation)

The codex session stopped at the quota ceiling after the battery reached
94% and before this section was written.  Completed here:

- Classification re-run from the raw record reproduces
  `classification.json` byte-for-byte (`HELD_R300_FIRST_W_LEAF_U_DIFFERENCE`).
- The five round-300 citation-map entries now pin their enclosing
  SUBROUTINE (they were text-and-line only).
- Citation gate at `44a62f5eb`: round-300 receipt PASS (5 citations, each
  at its pinned subroutine); default receipt PASS (274); a line-shift plant
  on the `divhor.f90` range and a routine plant on the `sshwzv.f90` range
  both fail the gate (exit 1, `SYMBOL-NOT-AT-LINE` / `ROUTINE-CONTEXT-MISMATCH`).
- Focused tests (round-300 split + citation gate): `20 passed`.
- Fidelity battery (`tests/ocean/fidelity`, `-n 12`, run in four quarters
  because the whole battery exceeds the 10-minute synchronous limit under
  load ~24): 3,605 passed, 7 skipped, 8 failed.  The 8 are exactly the
  pre-existing reds of round 299's battery (SI3 scalar-math gate, worktree
  stamp, round-35 stamp scope, round-129 spread floor, two round-33 stale
  anchors, round-209 OMT-1 module set, round-228 citation map); none names
  a round-300 file.  One GYRE end-to-end plant test
  (`phase3_stage_sweep_gate::test_planted_stage_control_exits_nonzero_end_to_end`)
  did not finish within 10 minutes even alone; it exercises no file this
  round changes.  The ORCA2 grid-ladder straggler passed in isolation.
- GYRE ten-step trajectory and 30-day member: **not required** — this round
  changes no file under `packages/` (no model change).
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Review response (independent Claude review: HOLD)

Evidence: `orca2_rounds/round300/` — reviewer scratch in `review_scratch/`
(re-run here, numbers below re-derived from its arrays, not copied), the
r300d re-measurement at tree `374222def1d9` (`raw_measurement_r300d.json`
SHA-256 `c8650266b0feb50166338a97c1e2659f2e75920bc1ab787097400b60d7e446f8`,
`classification_r300d.json` SHA-256
`6099e505a37196a6e99d2dfc30b50ef032705def24c62bcedd735144a45e3b92`,
status `HELD_R300_FIRST_W_LEAF_U_DIFFERENCE`), and `test_breaks_r300d.log`.
The re-measurement's two labels are array-identical (every row, arm and
replay); every row except the corrected ratio row equals the original table
to the last printed digit.

### Finding 1 — stage-1 U transport (the round's main finding)

Same quantity on both sides: `e2u*e3u(Kmm)*(uu+zub)` in m3/s at the stage-1
entry level on the east face.  NEMO builds it at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:290`, records it at `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:599`, and
differences it in `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/divhor.f90:132-139`; the card side is the stage
transport with the halo column dropped.

| Quantity | Value | Label |
|---|---|---|
| Alignment: unshifted vs best one-cell shift, interior RMS | `8.44e4` vs `3.50e5 m3/s` (4.1x better) | CONFIRMED |
| Interior slope / correlation, card on NEMO | 0.996 / 0.990 | CONFIRMED |
| Fold band, cell `[147,47,25]`: card vs NEMO | `7.726e7` vs `1.346e5 m3/s` | CONFIRMED |
| Fold band: max / RMS of card-NEMO; NEMO's own max / RMS | `7.91e7` / `6.82e6`; `1.07e6` / `8.21e4 m3/s` | CONFIRMED |
| Fold band share of the squared domain difference (4,164 of 430,552 cells) | 98.5% | CONFIRMED |
| Interior RMS difference relative to NEMO's interior RMS | 14.0% | CONFIRMED magnitude; cause (six steps of independent trajectory) PLAUSIBLE |

These numbers are computed on the **independent** label's operands; the
**given NEMO's entry** label's operands are array-identical in the r300d
re-measurement, so they hold for both.  So the U-difference row's
`1.08e6` RMS is a fold-band defect, not a domain-wide one.

### Correction 2 — the ratio row compared two time levels

NEMO's stage-1 W uses r3t(Kaa) = 2/3 r3t(Kbb) + 1/3 r3ta: the np_HYB
default at `ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:51`, applied at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stprk3_stg.f90:179`, built from this step's barotropic sea level.  The
carried after-sea-level slot feeds only stp_2D's first `wzv`
(`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stp2d.f90:150-156`) and `dyn_spg_ts` overwrites it at
`ORCA2_OMIP_L4_R299W7/BLD/ppsrc/nemo/stp2d.f90:303`.  The record's `w_r3aa` equals `ext_r3t` bit for bit.
The card side's delta spans the full step to sea level N+1, so the committed
row compared a full-step delta with a one-third-step delta: the card/NEMO
delta ratio has median 2.9999 (16,375 cells with |NEMO delta| > 1e-7), and
the old row's max `5.075e-2` and RMS `1.364e-3` are about twice NEMO's own
stage-1 signal (max `2.523e-2`, 2xRMS `1.347e-3`).

Rebuilt at matched time level (NEMO delta x 3, the clock ratio), the
residual is max `9.4070653952792217e-3`, RMS `2.1140007761263205e-4`
(fold band max `1.92e-3`, interior max `9.41e-3`), against NEMO's own
full-step signal max `7.57e-2`.  This is its own row in the table above.
CONFIRMED (re-measured).  Consequences:

- The stretch-rate row was already time-matched (each side uses its own
  clock) and is unchanged.
- The cumulative arms are unchanged: with NEMO's U and V substituted, the
  card's own full-step stretch leaves `7.44e-4 m/s`, which is a matched
  comparison.  So a second member exists, but it is the **step's sea-level
  change**, not a "ratio/clock pair": the clock is a time-level
  representation.  R300-P4 stays REFUTED with that reason; the earlier
  attribution is withdrawn.

### Finding 3 — the round-300 tests were vacuous; now they bite

The all-zero W test is replaced by a bit-exact NumPy known answer on
non-zero operands (dry cells, dry bottom level) passed as traced `jit`
arguments, at NEMO's stage clock; a second test pins the time-matched ratio
helper (not the measurement's call of it: switching the time-matching off at
that call still passes, per the independent review).  Each planted break fails and was restored (`git status --porcelain`
empty after each):

| Break | Result |
|---|---|
| (i) west neighbour taken from the east | 1 failed, 3 passed |
| (ii) stretch removed from the W recurrence | 1 failed, 3 passed |
| (iii) NEMO clock dt instead of dt/3 | 1 failed, 3 passed |
| (iv) ratio row not time-matched | 1 failed, 3 passed |
| restored | 4 passed |

### Finding 4 — private-import ratchet

Commit `5073c0601` promotes the EEN ratio fold exchange to a public name (no
behaviour change; allowlist stays empty).  The ratchet passes (see gates).

### Decision 132 evidence (reviewer-measured scratch, re-analysed here)

Switching the ORCA2 card to the carried after-sea-level slot
(`rk3_extrapolated_carried`) moves the kt=7 operands (U transport by up to
`1.254e4 m3/s`, V by `2.541e4`, W by `2.76e-5 m/s`) but changes every
whole-domain max and RMS distance to NEMO by at most 0.040% (independent
label only); the interior RMS of the step-entry sea-level difference moves
0.046%.
The card's selection is not changed.

### Review-response gates

All run with `JAX_ENABLE_X64=1` and the landing script's `PYTHONPATH`.

- Citation gate at `3493522ed`: this receipt PASS (11 citations, 0
  failures, 0 failing map entries); default receipt PASS (274); a line plant
  on the stage-1 Kaa blend and a routine plant on the `dyn_spg_ts` call both
  fire (exit 1).
- Private-import ratchet: `6 passed`.
- Focused tests (round-300 split, citation gate, G1 grid ladder, ORCA2
  parallel receipt gate): `104 passed`.
- Remaining landing-gate unit tests: 118 PASSED, 0 failed under `-n 12`;
  the one that did not finish within 10 minutes under load (flat-bottom card
  bit-identical under the mask-rank arm) passes alone: `1 passed`.
- GYRE ten-step / 30-day member: not run here; the only package change is
  the symbol rename (no behaviour change), and the landing script runs the
  DINO month gate because a package file changed.
- codex review: paused (quota).

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 126 remains pending and is not re-raised; runoff is unchanged.
- Sea ice and the ORCA2 card's `unmeasured_features` remain unchanged.

## OPEN

1. Walk the recorded `pFu(ji)-pFu(ji-1)` boundary one producer upstream,
   scored in the fold band first (the interior is non-bit from trajectory
   divergence alone):
   split NEMO's stage-1 U transport product into metric, live U thickness,
   corrected stage U velocity, external correction, and mask, at matched
   program points under both labels.
2. The second member is the step's sea-level change (time-matched ratio
   residual, interior max `9.41e-3`); split it after U, at matched time level.
3. Keep the complete fold unit private and atomic.  Nothing lands until its
   remaining owner is cited and OMT-4 completes under both labels.
