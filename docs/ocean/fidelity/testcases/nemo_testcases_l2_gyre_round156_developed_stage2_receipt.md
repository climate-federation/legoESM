# NEMO testcase L2 GYRE Round 156 receipt — developed stage-2 velocity split

Date: 2026-09-22  
Status: **HELD** — no physics or configuration changed.  The external
barotropic half of NEMO's compiled stage-2 program carries **none** of the
developed stage-2 velocity difference: installing NEMO's own recorded depth
mean removes −0.036% of it by rms and −0.017% by maximum, i.e. it is very
slightly worse.  The magnitude is owned by the INTERNAL 3-D half — the stage
right-hand side at N+1/3 and the thickness-weighted assignment — plus one
bounded external term the record cannot vary, `r3u(Kaa)`, whose first-order
carry is about 2.4e-5 of the difference.  The internal half's operands do not
exist at day 180 in any admitted record.  An operator-run
acquisition is requested.

## Frozen scope

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round156.md`, committed as
`5d6ae27a93b936cd62dacc0473543c126fb58807`, before any stage-2 measurement
ran.  The authoritative measurement commit is
`167d088e9e686c87dfaafe6c692bcbfe642edd3b`, on a clean tree; it reproduces the
same table value for value at the three earlier commits that measured it
(`63728b2a61e9accb46be34f2ceb0f1979a2934af`,
`dd75894aaa78d746404b2f4fe1ae51ba9f62a555`,
`852bc269efa8ddd78bb6de1ed53d808cb7f8caf7`).  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round156/`; the measurement
is `developed_stage2_split.json`.  Commits after it are this receipt and the
campaign-state line, neither of which moves a number.

No physics, configuration, default, carried state, restart schema, stabilizer,
year harness, reconciliation gate, freshwater pair or #1484 guard changed.  No
configuration choice was made.  Without a source-exact candidate there is no
ladder, month, year, DINO, LOCK_EXCHANGE or OVERFLOW candidate arm.

## Compiled program

Cited from the record's own build.  Stage 2 binds `Kbb = N`, `Kmm = N+1/3`,
`Kaa = N+1/2`, and stage 3 reads that `Kaa` field as its `Kmm`, so the field
round 155 named the magnitude owner is the output of this program.

The EXTERNAL half sets the stage's `Kaa` fields — under the active `np_HYB`
arm `ssh(Kaa)` is a half-sum, `uu_b(Kaa)` is the final external-mode velocity
and `r3u(Kaa)` is a half-sum — at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:217-256`, and
closes with the barotropic correction
`zub = uu_b(Kaa) - SUM(e3u_3d*uu(Kaa))*r1_hu_0` at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:753`, applied at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:772`.

The INTERNAL half is the stage right-hand side at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:400-497`, whose
three operators are the pressure gradient at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:475`, which
OVERWRITES the right-hand-side slot, the vorticity at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:485` and the
advection at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:497`; and the
thickness-weighted assignment at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:697-699`.

Stage 2 carries NO lateral diffusion and NO implicit vertical part: the
lateral mixing call at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:732` sits inside
the stage-3 arm and the vertical one at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:745` is guarded
on stage 3.  That is read off the compiled guards, not inferred.  The lateral
momentum diffusion round 149 made bit-exact is the one inside the
two-dimensional slow-forcing routine, and it reaches this field ONLY through
the external half — the half this round measures to carry none of the
difference.  That is consistent with round 149's landing moving day 240 by
2.2e-8 K on 1.64e-2 K.

## Method

One production step from NEMO's admitted day-180 entry through
`LatLonCGridOceanModel.step` under production JIT, reusing the round-155
transport observer unchanged; the observer's full returned state is
bit-identical to an ordinary step.  The split calls the model's OWN shared
transcription of the compiled correction statement,
`rk3_stage_barotropic_correction` in
`packages/ocean/legoesm/ocean/dynamics/barotropic_common.py`, driven with
NEMO's recorded `uu_b(Kmm)`, `e3u_0`, `r1_hu_0` and `umask` — not a copy of
the arithmetic written inside the instrument, which is the blocker an
independent review raised in round 155.  Arms are an isolated JIT of that
shared statement and are never relabelled as production.  Scoring is over the
17,400 active U faces; the rms ranks and the maximum is reported beside it.

The whole round-155 compiled-order table and operand attribution were re-run
at this round's commit and reproduce value for value, including the
calibration arm in which NEMO's own operands rebuild NEMO's transport BIT
(0 of 21,780 cells) and the `uu(Kmm)` arm at 95.21% of the transport rms.

## The split

| arm | active unequal / 17,400 | active max abs | active rms | rms removed |
|---|---:|---:|---:|---:|
| calibration, NEMO's own field | 12,116 | 1.474514954580286e-17 | 1.7851305060605976e-18 | — |
| production baseline | 17,400 | 1.30926020461275e-6 | 2.7410518419398658e-8 | 0 |
| NEMO's depth mean installed | 17,400 | 1.3094852667761003e-6 | 2.7420273097734563e-8 | −3.558735441137203e-4 |
| same, unmasked reference weight | 17,400 | 1.3094852667761003e-6 | 2.7420273097734563e-8 | −3.558735441137203e-4 |

The calibration row is the decomposition's own floor: re-applying NEMO's
correction statement to NEMO's own already-corrected velocity moves 12,116
faces by at most 1.47e-17 m/s, eleven orders below the difference being
split, so every share above is safe.

The masked and unmasked reference-weight arms agree to every digit, so
legoESM's dry-face velocity contributes nothing to the column sum and the mask
inside NEMO's weight is inert here — measured, not argued.

**Known answer, and it is not circular.**  What the substitution removes at
each level is the depth-mean error the stage installed, so it must reproduce
the separately recorded barotropic-velocity row of the same walk — a row built
from an operand the substitution never touches, legoESM's own `uu_b(Kmm)`.  It
does: removed maximum 6.13420314543589e-10 m/s against the recorded
`uu_b(Kmm)` row 6.134203041352482e-10 m/s over 580 of 580 active columns,
disagreeing by 1.3010426069826053e-17.  The instrument reproduces a number the
walk already knew, to the calibration floor.

**Where it lives.**  The difference is surface intensified and the
substitution changes the profile nowhere.  Active rms by level, baseline then
after the substitution: level 2 1.0564807016308022e-7 / 1.0568239238668932e-7,
level 3 6.367421195247725e-8 / 6.370577130409173e-8, level 4
4.8097380763602544e-8 / 4.810955621715433e-8, level 5
4.7572157548957326e-8 / 4.756676898352136e-8, level 6
3.457216795018307e-8 / 3.4542679609543677e-8.

## Predictions and verdict

1. The decomposition's calibration is below 1.0e-15: **CONFIRMED**,
   1.474514954580286e-17 m/s.
2. The production `uu(Kmm)` row reproduces round 155 exactly: **CONFIRMED**,
   17,400 of 17,400 active faces, maximum 1.30926020461275e-6 m/s.
3. Installing NEMO's own depth mean removes less than 1% of the active rms:
   **CONFIRMED**; it removes −0.036%.  Registered against this prediction, as
   the independent review pointed out: it could hardly have failed, because
   round 155's own table already carried the recorded depth-mean row at
   6.134e-10 against a 2.741e-8 rms.  What the arm adds is not that number but
   prediction 4's control and the scope statement below; and the removed part
   is a thickness-weighted column mean while the score is an unweighted face
   rms, so the SIGN of −0.036% carries no meaning and only its magnitude does.
4. The removed depth-mean error reproduces the recorded `uu_b(Kmm)` row to
   within the calibration floor: **CONFIRMED**, disagreement 1.30e-17 against
   a recorded row of 6.13e-10.
5. The one-ULP plant on NEMO's recorded `uu_b(Kmm)` moves the reprojected
   velocity and exits nonzero: **CONFIRMED**.  A one-ULP change at the active
   column `[1, 2]` leaves the reprojected row's maximum and cell count
   *identical* — which is exactly why the first version of this plant reported
   that nothing moved — and changes the fingerprint of the whole reprojected
   field from `5d58e75b…` to `7560910d…`.  It prints
   `STATUS PLANT-FIRED: stage2-uu-b-ulp` and exits 1.

## What is owned, and what is still open in the external half

The substitution varies the external half's FINAL statement, the correction
at `GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:753` applied at `GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:772`.  The external half writes THREE fields at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:217-256`, and the other two reach the velocity by a different route:
`ssh(Kaa)` only through `r3u(Kaa)`, and `r3u(Kaa)` — which is stage 3's
`r3u(Kmm)` — is the divisor of the INTERNAL assignment at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:697-699`, not of the correction, so this arm does not vary either.  `r3u(Kaa)` is measured non-bit at a RELATIVE maximum of
6.522560269672795e-13 (the round-155 row, reproduced this round).  To first
order its carry is that relative difference times the velocity, about
2.4e-5 of the measured rms — a bound read off a measured operand, labelled
**PLAUSIBLE**, not a substitution arm.  So what survives this arm is the
internal half PLUS that bounded term, and round 157's record settles it by
substitution rather than by the bound.

## First non-bit statement

None can be named, and the reason is a missing record, not a missing
measurement.  Everything the admitted day-180 record can substitute in the
stage-2 correction has been substituted and none of it carries the magnitude:
`e2u`, `umask` and `r1_hu_0` are bit-exact, `e3u_0` is bit-exact on every
active face, `uu_b(Kmm)` is non-bit at 6.13e-10 and measurably inert, and the
correction statement itself is exonerated by its own calibration.  Apart from the bounded `r3u(Kaa)`
term registered above, the 1.3093e-6 m/s difference is born inside the stage
right-hand side at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:400-497` and the
assignment at
`GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:697-699`, whose
operands — the stage-1 velocity at N+1/3, the three right-hand-side snapshots,
the N+1/3 free-surface ratios, the density anomaly and the cross-level
velocity — every existing build writes only at the first step.

This is a one-step magnitude at the developed day-180 entry.  No day-240 carry
is claimed for any operand, because no candidate exists to score.

The campaign headline rows remain inherited and unmodified: kt2 T/S at the
bar, kt2 U/V approximately 2.7377e-12 / 3.2849e-12, kt3 T approximately
8.60e-7 K, day-30 6.888194e-5 K, day-240 1.644671864e-2 K, day-360
1.122345086e-2 K.  ORCA2 is **UNMEASURED-WITH-SPEC**: repeat this stage-2
velocity split on the ocean-only ORCA2 card before transferring the verdict.

## Acquisition requested

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round156_developed_stage2/run.sh`,
operator-run, target `GYRE_OMIP_L2_P3_SM_R156ST2`, output
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round157/oracle_developed_stage2`.
It records nineteen operand groups at step 1081, stage 2, through an additive
write-only module: the entry right-hand side before the pressure gradient
overwrites it, the snapshots after the pressure gradient, the vorticity and
the advection, the before and N+1/3 velocities, the three free-surface ratios
and the stage time step, the density anomaly and cross-level velocity the
operators read, the raw and corrected after-velocities, the written
barotropic correction, the external-mode velocity, and the two flags that
select which compiled arm the assignment takes.  The patch removes no source
line.

Preflight, run here: `SYNTAX_PROOF_PASS l2_r156_stage2.f90 stprk3_stg.f90`,
`LAYOUT_ROUNDTRIP_PASS 4 groups, 2816 bytes`,
`ROUND156_DEVELOPED_STAGE2_PREFLIGHT_READY`; and the layout plant printed
`STATUS PLANT-FIRED: layout rhs_entry` and `STATUS PLANT-FIRED: layout
after_adv`, and exited 69.

**A defect the preflight caught, before any NEMO time was spent.**  The first
writer decided whether its file was open by testing the SIGN of its unit
number, and Fortran's `NEWUNIT` hands back a NEGATIVE unit, so every group
returned early and the record would have been an 80-byte header.  Syntax,
layout and the run script's own greps all passed on it.  It was found by
compiling the writer against small stubs and parsing its bytes with the
admission gate's own reader.  Openness is a logical flag now, and that
round-trip is a fail-closed preflight step rather than a one-off: it refuses
unless the reader recovers the column-major mapping, the pair order, the
interior extents of the barotropic correction and the two scalars.

Admission follows operator note AS: the inherited-stream check is waived and
recorded as waived; passivity is judged by NEMO's own restarts staying
byte-identical to the un-instrumented Round-132 daily reference, by the
requirement that the record's stage-2 after-velocity be byte-identical to the
admitted Round-154 record's stage-3 `uu(Kmm)` — the same array by NEMO's own
binding, and the only check that speaks for step 1081 itself — and by the
in-run calibration in which the recorded corrected velocity is rebuilt from
the recorded raw velocity and the recorded correction.  The
thickness-weighted assignment rebuild is REPORTED and explicitly NOT gated,
because its Fortran association may be contracted by the compiler and this
campaign proves that statement under production JIT rather than in a reader.

## Controls, review and tests

The instrument's own controls, all inside the run: the transport observer's
returned state is bit-identical to an ordinary step; the unsubstituted
transport arm is byte-identical to the production row; the fully substituted
arm rebuilds NEMO's transport bit for bit; the split's calibration arm and its
known-answer cross-check are the two rows above.

The stage-2 plant is controlled on the whole reprojected field, not on its
maximum.  The first version compared maxima and reported that a one-ULP change
to NEMO's recorded depth mean had moved nothing — true, and the wrong test: it
moves about thirty cells by about 1e-16, which no maximum over 17,400 faces
can see.  That is registered here rather than quietly fixed.

The new split test pairs a pure depth-mean difference, which the substitution
must remove, with a pure deviation difference, which it must not, and it was
shown to FAIL when the substitution is dropped from the arm
(`assert 6.0000000000060005e-06 < 1e-15`).  The new admission-gate test admits
a self-consistent synthetic record and refuses a broken corrected velocity, a
moved restart and a missing group, and drives all four plants.

Independent review: the codex autopilot is paused, so codex was NOT invoked —
`independent review not run (codex paused)`.  A separate Claude reviewer with
no part in the implementation read the whole diff and returned **DO NOT SHIP**
on the first version, with three blockers, verbatim:

> B1 — The substitution varies ONE of the external half's three outputs, but
> the claim covers all of them.
> B2 — No check that the new record's step-1081 state is the same state the
> walk already uses.
> B3 — A layout guard that can go vacuous, in the script that authorises the
> run.

All three are closed, and the review's own suggested settlements were used.
B1: the arm's docstring and this receipt now scope the claim to the external
half's final statement and register `r3u(Kaa)` as a bounded, unvaried term
(section above); the review's arithmetic — `6.52e-13` times the velocity,
about `2.4e-5` of the rms — is reported as its bound, and round 157 settles it
by substitution rather than by that bound.  B2: the admission gate now
requires the new record's stage-2 after-velocity to be byte-identical to the
Round-154 record's stage-3 `uu(Kmm)`, which NEMO's own binding makes the same
array, with its own plant; that is the check that sees a writer perturbing
step 1081, which the restart comparison of steps 180 to 1080 cannot.  B3: an
empty line number compares greater than nothing in the shell, so two of the
six layout endpoints were never really required; they are now, and the layout
plant exercises both guarded arms (`STATUS PLANT-FIRED: layout rhs_entry` and
`... after_adv`).

The review also confirmed, independently, that the arm calls the shared helper
rather than a copy, that the known-answer control is NOT circular, that the
pytest arm is non-vacuous in both directions, that every compiled citation
above resolves at its cited line, and that the Fortran writer is passive with
`zub` captured live at its real interior extents.  Its non-blocking notes are
closed too: the record gains the two flags that select the compiled arm, the
gate's docstring no longer claims the rebuild is order-independent (it is safe
because `umask` is exactly 0 or 1, which makes the product exact), and the
`jpk` versus `jpkm1` window assumption is named.

To reproduce: audit the daily record with
`nemo_testcase_l2_gyre_round131_daily_record_gate.py --audit --expect-commit
<tip> --daily-root phase3/round132/oracle_daily_restarts --ready-stamp
phase3/round132/oracle_daily_restarts/round133_daily_restarts.stamp`, then run
`nemo_testcase_l2_gyre_year_owners.py --developed-transport-walk` with that
audit, `--daily-record-root` on the same root and both record roots on
`phase3/round154/oracle_developed_transport`.

The six-file push gate, the year-owner harness tests and this round's two new
test files were run together and reported exactly:

> 178 passed in 1004.83s (0:16:44)

The citation gate on this receipt reported `PASS` with 10 citations, zero
failures, zero unmapped citations and zero map entries failing audit; its
shifted-line plant on the barotropic-correction citation exited 1 with
`SYMBOL-NOT-AT-LINE`.  The whole map was audited separately and is green at
274 citations on the cumulative receipt.

## OPEN — Round 157

The stage-2 velocity difference is owned by the internal 3-D half of NEMO's
stage-2 program, and the operands of that half do not exist at day 180.  The
acquisition above is the round's deliverable; round 157 begins with it.

1. The operator runs
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round156_developed_stage2/run.sh`.
   Admit the record with the committed gate
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round156_developed_stage2_gate.py`:
   `STATUS PASS` plus the four plants, each printing its own
   `STATUS PLANT-FIRED` line.  If the restarts move, the writer is not
   passive and must be fixed under a NEW target name, which is the only
   refusal operator note AS leaves.
2. Then walk the stage-2 right-hand side in compiled order under production
   JIT from the same day-180 entry: score the entry slot, then the snapshots
   after the pressure gradient, the vorticity and the advection against
   NEMO's recorded ones, and name the first non-bit family with the magnitude
   it carries.
3. Keep the calibration discipline rounds 155 and 156 established.  Before
   naming any statement, show that legoESM's transcription rebuilds NEMO's
   output bit for bit from NEMO's OWN recorded operands.  The new record makes
   that available for the thickness-weighted assignment at
   `GYRE_OMIP_L2_P3_SM_R154TRPWALK/BLD/ppsrc/nemo/stprk3_stg.f90:697-699`
   first, because it carries every one of its operands; do that one before
   the three operator families, because a non-bit assignment would make each
   family's row inherited rather than owned.
4. The record also carries the stage-1 output velocity at N+1/3, so a family
   that is non-bit can immediately be classified as OWNED or INHERITED
   instead of being walked blind.
5. Only a single source-exact production-JIT statement becomes a candidate,
   and it must pass the full Decision 43/45 gate — month, day 240, day 360,
   the kt2 T/S bar, first-over-bar, every moved row, DINO if the statement is
   shared, the generic NEMO-GYRE card, the LOCK_EXCHANGE and OVERFLOW tanks
   and the six-file push gate — before landing.  Rank by what it carries at
   day 240, not at day 180.
6. `un_adv` stays registered as the compiled-order first unequal input of the
   stage-3 transport and is still NOT a candidate: round 155 measured that it
   removes none of the transport difference, and this round measured that the
   whole external half it belongs to removes none of the velocity difference
   either.
