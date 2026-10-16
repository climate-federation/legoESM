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

The cited record is the compiled `ORCA2_OMIP_L4_R299W7` configuration.  It
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
| Kaa-Kbb ratio delta | 16,433 / 16,433 | 5.0750835119969487e-2 | 1.3644301289032852e-3 | `[54,35]` |
| stretch rate | 430,552 / 430,552 | 8.836174225055932e-5 | 8.009334080397122e-7 | `[147,48,27]` |
| W recurrence | 430,529 / 430,552 | 1.9152110475369796e-1 | 1.5744398704267452e-3 | `[147,48,0]` |
| area product | 430,529 / 430,552 | 1.4843986124373436e9 | 1.222542183061254e7 | `[147,48,0]` |

Replacing candidate U transport alone leaves W maximum error
`0.19217452883025382 m/s`; replacing U+V leaves
`0.0007441398648953929 m/s`.  Area, live Kmm thickness, and mask do not
close that residual.  The replay becomes exact only when the recorded
Kbb/Kaa ratios and stage clock are added as their coupled pair.  Therefore
the preregistered claim that the complete divergence alone closes W is
**REFUTED**: the first boundary is U, while the W result has an independent
ratio/clock member.

## Prediction ledger

| ID | Verdict | Result |
|---|---|---|
| R300-P1 | **CONFIRMED** | Admission, provenance, both labels and all controls pass. |
| R300-P2 | **CONFIRMED** | Candidate and oracle replays pass in W-equivalent units; the raw area-product row remains reported. |
| R300-P3 | **CONFIRMED** | U difference is the first non-bit boundary on both labels. |
| R300-P4 | **REFUTED** | Divergence substitution alone does not close W; the Kbb/Kaa/clock pair is independently required. |
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

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 126 remains pending and is not re-raised; runoff is unchanged.
- Sea ice and the ORCA2 card's `unmeasured_features` remain unchanged.

## OPEN

1. Walk the recorded `pFu(ji)-pFu(ji-1)` boundary one producer upstream:
   split NEMO's stage-1 U transport product into metric, live U thickness,
   corrected stage U velocity, external correction, and mask, at matched
   program points under both labels.
2. Keep the Kbb/Kaa/clock pair coupled; its half-states are not causal claims.
3. Keep the complete fold unit private and atomic.  Nothing lands until its
   remaining owner is cited and OMT-4 completes under both labels.
