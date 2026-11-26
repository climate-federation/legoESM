# ORCA2 round 270 — G1 transfer landing under Decision 96

Date: 2026-10-13  
Lane base: `42b598288`  
Labels: **independent** and **given NEMO's entry**, never mixed  
Verdict: **LANDED**

## 1. Frozen unit and oracle statements

The frozen preregistration is
`PREREG_nemo_testcases_l4_orca2_round270.md` at `08f1ff877`.  Round 270
restores the exact six selector-gated statements measured in round 269; all
eight implementation, card-gate and unit-test blobs are byte-identical to
candidate commit `526346399`.  It does not re-derive the geometry-ladder work.

The compiled ORCA2 record build says:

1. EEN constructs the U/V transports and consumes their fold-image row in the
   triads at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:784-785` and
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:790-802`;
2. the split-explicit midpoint face depths use the adjacent T-point sea
   surfaces at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:509-520`;
3. stage 1 constructs SSH as NEMO's written two-product sum at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178`;
4. the external-substep length is the true quotient at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1077`;
5. the flux-form velocity bracket is multiplied by the mask-safe inverse
   exit depth at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:655-718`;
6. the loop-entry face depth is reference depth times one plus the live ratio
   at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:361-364`.

These are transcriptions, not stabilisers.  No deck, carried state, initial
state, sea-ice selector, or `unmeasured_features` entry changed.  OMT-0
selects the two flux-form-only statements.  OMT-1..4, rung 0 and shipped rung
10 select the four shared/vector statements and leave the two flux-form-only
selectors off.  GYRE, DINO and the tank cards resolve all six selectors off.

## 2. Decision-96 census

The round-269 measurements remain admissible because the selected candidate
blobs are exact and their SHA-256 files are retained under `round270/`.
Round 270 changes the classifier to implement Decision 96 as written: strict
RMS majority toward NEMO, no exact-row loss, and the first non-bit row toward
or equal.  The final stage-3 SSH maximum remains a reported diagnostic, not a
new veto.

| card / label | rows | RMS toward | RMS away | RMS equal | exact lost | result |
|---|---:|---:|---:|---:|---:|---|
| OMT-1 independent | 160 | 133 | 7 | 0 | 0 | PASS |
| OMT-1 given NEMO's entry | 160 | 133 | 7 | 0 | 0 | PASS |
| OMT-2 independent | 200 | 165 | 15 | 0 | 0 | PASS |
| OMT-2 given NEMO's entry | 200 | 165 | 15 | 0 | 0 | PASS |
| OMT-3 independent | 200 | 162 | 18 | 0 | 0 | PASS |
| OMT-3 given NEMO's entry | 200 | 162 | 18 | 0 | 0 | PASS |
| OMT-4 independent | 200 | 160 | 24 | 0 | 0 | PASS |
| OMT-4 given NEMO's entry | 200 | 160 | 24 | 0 | 0 | PASS |
| rung 0 independent | 200 | 159 | 25 | 16 | 0 | PASS |
| rung 10 given NEMO's entry | 200 | 161 | 24 | 0 | 0 | PASS |

For rung 0, all 200 rows are bit- or score-touched.  The first non-bit row is
equal.  The registered final stage-3 SSH maximum changes
`0.3840334615160934 -> 0.3840334668286476 m`, a `5.312554174352613e-9 m`
increase (`1.38335710471156e-8` relative); it is not a vote and not a veto.

Shipped rung 10 required a new same-tree selector-off control because it
executes four transferred statements.  Its fresh **given NEMO's entry**
candidate ladder completes kt=1..10.  The committed round-270 classifier
finds 185/200 touched rows: 161 toward, 24 away and 0 RMS-equal; 15 rows are
unchanged, no exact row leaves the bar, and the first non-bit row is equal.
Its planted exact-row loss exits 2.  This replaces any attempted comparison
to an older tree.

The matched stage-2 EEN result is preserved under both labels: after-VOR U
maximum `1.7082229745349855e-6 -> 1.2863307536456779e-8 m s-2`, with selected
RMS `1.2191816847794566e-10 m s-2`; the accumulator self-replay remains
`3.3881317890172014e-21 m s-2`.

## 3. Shared-card and long-run gates

- GYRE's kt=1..10 residual artifact is byte-identical to the certified
  digest `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
- The CPU-only 360-day GYRE member is byte-identical at the registered
  checkpoints: day 30 `3c0602babb535aac...`, day 240
  `2e2b895c72b3dd02...`, and day 360 `5af258eff135981f...`.  Therefore the
  certified T RMS values remain exactly `2.3432419318363155e-06`,
  `6.5816987106668941e-05`, and `5.4077212586815052e-05 K`.
- DINO's CPU-only from-rest month passes at `2.056821682e-03 K` against the
  `2.244317642e-03 K` bar.  Its day-30 snapshot digest is
  `d2e254e5e9d3f9dbddc53c4c9676ea2d992af232fda58ac51cfc1710cbd46b55`.
- The generic-card, UP3-scope, lock-exchange/overflow and DINO focused gate
  battery passes 61/61.  The exact landing/push battery passes 128/128 in
  `529.38 s`.

Thus GYRE, DINO and tanks are unchanged where required.  No NEMO acquisition
was needed.

## 4. Full battery, controls, citations and review

The required `tests/ocean/fidelity -n 12` battery was run once.  It collected
3,466 tests, reached 99%, emitted six failure marks, and then stopped on its
last worker.  It was terminated rather than rerun.  Four emitted registered
reds reproduce serially with xdist disabled:

1. SI3 scalar-math `MY_SRC` provenance;
2. the stale GYRE round-129 certified-year harness stamp;
3. the 13 legacy allow-dirty scope offenders; and
4. the 13 legacy unstamped report emitters.

The two late-worker cases match the previously isolated pair: the GYRE
round-179 source-layout plant passes alone in `0.39 s`; the generic stage
sweep's `test_prediction_plant_is_fail_closed` again remained in its long JAX
subprocess beyond ten minutes and was terminated (resource red, not a model
assertion).  No round-270 path appears in any reproduced offender list.

The cumulative citation audit passes with zero failures and zero unmapped
citations after the required SequenceMatcher re-anchor.  The classifier's
planted exact-row loss refuses.  The round receipt citation gate and its
shifted-source plant are recorded after this receipt is committed.

The required separate `codex exec --sandbox read-only` review was attempted.
It failed before reading the diff because the in-process app-server could not
initialize on the read-only filesystem.  Therefore **independent review
unavailable in-sandbox**; the retained log digest is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

## OPEN

The transferred EEN fold-image statement strictly improves but does not close
the stage-2 EEN residual.  Resume the source-ordered interior leaf replay on
the landed tree: `ff_f`; relative circulation; `r1_e1e2f`; live
`e3f_0vor*(1+r3f)`; `zwx/zwy`; four triad sums; final metric and accumulator
addition.  The preregistered first candidate remains the live F-point
thickness over ORCA2 partial cells.  Use the production D114 operand route and
matched post-association program point; no acquisition or user decision is
currently needed.
