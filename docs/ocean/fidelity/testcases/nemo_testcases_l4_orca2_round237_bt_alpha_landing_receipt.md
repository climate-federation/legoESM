# ORCA2 round 237: required barotropic-filter alpha landing receipt

Date: 2026-10-11  
Status: **LANDED** (Decision 115, option 1)  
Scope: ocean only; no SI3 selector, implementation, or `unmeasured_features`
entry changed.

## Claim and source

This round removes the shared `0.07` library default from NEMO's AB3/AM4
barotropic time filter.  Every NEMO-derived card must now state the deck's
`rn_bt_alpha`; the constructor and the card registry refuse a missing or
non-finite value.  ORCA2 and OMT state `0.09`, GYRE/VORTEX/LOCK state `0.07`,
and the box-filter OVERFLOW and DINO cards state their inert deck value `0.0`.

The executed ORCA2 statement is the half-step SSH interpolation in
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:604-612`, whose
four weights are formed from `rn_bt_alpha` by `ts_bck_interp` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:1522-1557`.
The OMT-4 deck states `rn_bt_alpha = 0.09` in
`orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2/namelist_cfg:383-387`.
This is a deck value, not a new choice or stabiliser.

## Frozen prediction disposition

The preregistration was commit `f8a5a652b` and preceded all new measurements.

| Prediction | Result | Disposition |
|---|---:|---|
| P1 required registry and refusal controls | all NEMO cards explicit; missing/wrong/non-finite controls refuse | CONFIRMED |
| P2 OMT-4 `eta_pgf` | `8,794/13,320 -> 0/13,320` unequal; max `3.1271104752883805e-3 -> 0 m` | CONFIRMED |
| P2 weights | `[0.7125337174, 0.2525882038999999, 0.03722244, -0.0023443612999999985]`, bit-exact | CONFIRMED |
| P3 cards retaining their deck value | GYRE ladder/year byte-identical; filter-1/2 cards explicitly retain inert values | CONFIRMED |
| P4 ORCA/OMT ladders | all majority-toward; no exact row lost; first debt toward or unchanged | CONFIRMED |
| P5 month boundary | OMT-4 completes 96; rung-0 refusal moves later, step `96 -> 100` | CONFIRMED |
| P6 citations/tests/controls | citation and planted controls fire; focused and full battery below | CONFIRMED |

The first alpha-gate execution indexed the four-element expected-weight vector
as if it were two-dimensional.  That was a gate defect, fixed before the
first scientific comparison; the failed log is retained and no prediction
was changed.

## Source-exact statement gate

`alpha_gate.json` reports `PASS_R237_BT_ALPHA`.  Replaying the recorded OMT-4
substep-3 inputs with `0.07` reproduces the candidate mismatch on 8,794 cells
at zero ULP.  Replaying with the deck's `0.09` reproduces NEMO on all 13,320
cells at zero ULP.  Three independent plants fire: a missing card field, an
ORCA value of `0.07`, and a one-bit coefficient perturbation.

## Ten-step ladder census

All numbers below are oracle-relative.  “Moved” means RMS changed; score-equal
bit movement is not counted as a vote.  Each comparison's exact-loss plant
fires.

| Card / label | Rows moved | RMS toward / away | max toward / away | exact rows lost | first debt |
|---|---:|---:|---:|---:|---|
| OMT-1, each of both labels (kt 1..8) | 155 | 155 / 0 | 138 / 17 | 0 | toward |
| OMT-2, each of both labels | 195 | 195 / 0 | 169 / 26 | 0 | toward |
| OMT-3, each of both labels | 195 | 195 / 0 | 168 / 27 | 0 | toward |
| OMT-4, each of both labels | 195 | 195 / 0 | 141 / 54 | 0 | toward |
| rung 0, independent | 195 | 195 / 0 | 128 / 67 | 0 | toward/unchanged |
| shipped rung 10, Decision-52 gate protocol | 195 | 181 / 14 | 119 / 76 | 0 | toward |

The two labels are kept separate in the JSON artifacts.  The rung-10 gate's
label describes its existing Decision-52 surface bridge; it is not mixed with
the hierarchy cards' independent label.

Selected endpoints:

| Row | Before | After |
|---|---:|---:|
| OMT-4 kt1 stage1 SSH RMS / max (m) | `6.238271974498368e-3 / 0.1310917645484672` | `6.230274897792635e-3 / 0.13030042988356552` |
| OMT-4 kt10 stage3 SSH RMS / max (m) | `2.4352931961325045e-2 / 0.3949686188367629` | `2.4101820284146314e-2 / 0.39483051836679883` |
| rung 0 kt10 stage3 SSH RMS / max (m) | `2.802652392771078e-2 / 0.42832517646246693` | `2.725209920277513e-2 / 0.3840334615161052` |
| rung 10 kt10 stage3 SSH RMS / max (m) | `2.7586453643536082e-2 / 0.30375337870920743` | `2.6902838239443144e-2 / 0.3022728545882475` |

## Month and preserved-card gates

- **OMT-4, independent:** entry T/S/u/v/SSH are exactly NEMO's own entry,
  and legoESM completes all 96 admitted steps without a non-finite value or
  runtime refusal (`PASS_R237_OMT4_MONTH_COMPLETE`).  The claim-label plant
  refuses.
- **Rung 0, independent:** entry remains bit-exact.  The live-W-thickness
  refusal moves later from the registered step 96 to step 100 (99 completed),
  so the change introduces no earlier boundary.
- **GYRE:** the certified 70-row ten-step ladder and its residual arrays are
  byte-identical.  Fresh day-30/240/360 snapshots are byte-identical with
  SHA-256 `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`,
  `2e2b895c72b3dd0218f22a06078d944cbf92c91f75493c7d4e21ddc7eb985abe`,
  and `5af258eff135981fa80bfb3d4c354f034f66fcde9d9c3712f1608aff88e37646`.
  Thus the pinned T RMS values remain exactly
  `2.3432419318363155e-06`, `6.5816987106668941e-05`, and
  `5.4077212586815052e-05 K`.
- **OVERFLOW/LOCK/DINO:** the cards explicitly preserve their deck values.
  OVERFLOW (`nn_bt_flt=1`) and DINO (`nn_bt_flt=2`) do not execute the
  AB3/AM4 coefficient branch; LOCK states the same `0.07` previously used.
  The card-registry/refusal tests and the DINO month result below gate these
  assertions.  The exploratory OVERFLOW barotropic report remains its
  pre-existing scientific `DEBT`; it is not presented as a new regression.
- **DINO month:** fresh CPU/fp64 production run gives day-30 wet three-
  dimensional T RMS `2.056821682e-03 K`, below the fixed
  `2.244317642e-03 K` bar.  The log and score JSON SHA-256 values are
  `7845ab42a9cc71e1a13fb4b3c91844c52c09e0fc6337777df74694d482940648`
  and `27aadf9d72631e3d1c234c2ffb488015443559cabf9f6e0733730ceab8607426`;
  the high-regression/NaN self-test fires.

## Verification and controls

- Focused alpha/card/source tests: 52 passed.
- `tests/ocean/fidelity -n 12`: 3,152/3,162 reported PASS before the
  stage-sweep prediction-control worker exceeded ten minutes without output.
  Its exact ID then passed alone in 1,083.81 s; the other nine queued IDs
  passed alone as one bounded set in 538.10 s.  Thus all 3,162 selected tests
  passed, with no failing ID hidden by the interrupted worker pool.
- Default citation gate: zero unmapped citations, failures, or map-audit
  errors.  The fake `stprk3.F90:186` citation plant refuses.
- Independent read-only review: **independent review unavailable in-sandbox**;
  `codex exec --sandbox read-only` failed before reading the diff because its
  in-process app-server could not initialize on the read-only filesystem.
  The exact stderr is retained in `independent_review.log`.

## OPEN

1. Round 238 lands Decision 114's exact ORCA geometry unit in its own commit:
   raw U/V face thickness through the existing resolved builder and NEMO's
   T/U/V/F fold-mask rules, with the full card and moved-row gates.
2. Round 239 re-scores the round-217 atomic V-transport unit with alpha and
   geometry both landed, including the Ve_rhs fold-pair discrimination,
   OMT-4/rung-0 months, and all preserved-card gates.
3. OMT-0's registered flux-form V-update debt remains out of this vector-form
   landing, exactly as Decision 109 ordered.
