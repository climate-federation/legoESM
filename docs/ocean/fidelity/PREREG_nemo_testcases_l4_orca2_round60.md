# ORCA2 round 60 preregistration — OVERFLOW downstream UP3 walk

Date frozen: 2026-09-28  
Base: `58f578850c23f9c11b4d1e5c046cdbabadd2e334`  
Cards: `OVERFLOW-zps`; the held shared UP3 statement is measured as a controlled arm  
Claim labels: OVERFLOW trajectory **independent**; statement records **given NEMO's recorded operands**

## Frozen scope and source order

Round 59 proved that NEMO's source-ordered UP3 U T-face flux closes from
282 / 17,000 unequal to 0 / 17,000 and leaves ORCA2 and GYRE byte-identical,
but it moves 31 OVERFLOW trajectory rows and violates the two-ULP bar on 20.
The candidate remains held.  This round does not retry its landing.  It walks
the first movement downstream from kt=3 stage-2 advection and names the first
statement that changes the candidate's direction against NEMO.

The authoritative compiled program is
`OVERFLOW_OMIP_L1_P3_R50PAIR/BLD/ppsrc/nemo/stprk3_stg.f90`.  Stage 2 records
the advection accumulator at :354-363; no executed tendency lies between it
and `pre_zdf` on this deck (:374-429); the QCO velocity update is :396-404;
and the reference-thickness barotropic replacement is :436-448.  Stage 3
adds lateral mixing at :412-429, calls the implicit vertical solve at :430-433,
and applies the same barotropic replacement at :436-448.  The compiled
`dynzdf.f90:139-170` gives the explicit update and implicit-drag entry before
its vertical matrix.  These are the only eligible boundaries, in that order.

The admitted round-50 self-describing record already contains stage-2
`after_adv`, `pre_zdf`, `raw_kaa`, `postbar_kaa`; stage-3 `after_adv`,
`after_ldf`, `pre_zdf`, `raw_kaa`, `postbar_kaa`; and the existing kt=4 entry.
The instrument must parse the record header and fields through the committed
round-50 reader, score the card's physical active U cells, stamp CPU/fp64/JIT,
and publish an array sidecar so base-to-candidate movement is bitwise gated.

Repository search found the existing narrow production observers
`expose_stage2_momentum_rhs`, `expose_stage2_raw_momentum`,
`expose_momentum_stage=2`, and `expose_stage3_momentum_rhs` plus the round-50
reader and round-51 kt=3 driver.  The new gate reuses those; it does not add a
second model operator, configuration selector, or model-file diagnostic.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R60-P1 | The round-50 kt=3 record remains admitted and the base kt=3 entry reproduces the admitted state. | Admission is `AT_BAR`; all five entry fields are bit-exact; a one-ULP plant makes exactly one scored cell refuse. | Any record/stamp/header/entry drift: stop and reconcile. |
| R60-P2 | The held UP3 arm first moves the live source walk at stage-2 `after_adv`. | Every earlier exposed boundary is unchanged and `after_adv` moves on at least one active U cell. | An earlier boundary moves or `after_adv` does not: round 59's attribution is refuted. |
| R60-P3 | Stage-2 `pre_zdf` is bit-identical to `after_adv` within each arm because no intervening branch executes. | Their uint64 payloads are array-equal in base and candidate. | Any bit differs: stop; an omitted executed statement exists. |
| R60-P4 | The first compensating owner is either the QCO stage update or the following barotropic replacement. | At the first of `raw_kaa`, `postbar_kaa` where the candidate/base movement shrinks or its oracle-relative error changes direction, the gate names that compiled statement and reports counts/maxima. | Neither boundary changes direction: carry the movement into stage 3 without naming an owner. |
| R60-P5 | If stage 2 does not own the reversal, the ordered stage-3 `after_adv`, `after_ldf`, `pre_zdf`, final/postbar, and kt=4 entry walk names the first reversal or proves it remains downstream. | Exactly one earliest boundary is selected mechanically from signed oracle-relative maximum and unequal-cell changes; all earlier rows retain direction. | A required NEMO boundary exists but has no production-bound observer: finish `STOPPED_FOR_RECORD` with a fail-closed additions-only acquisition. |
| R60-P6 | The measurement is passive and non-vacuous. | Every observer's ordinary final state is bit-identical to the unobserved production step; the sidecar hash binds; a real active-U one-ULP plant changes exactly one row/cell and exits nonzero. | Any observer moves production or the plant fails: instrument invalid. |
| R60-P7 | No physics lands this round. | Final `packages/` tree equals base; held UP3 and QCO/RK candidates remain absent; GYRE/ORCA2/ice are unchanged. | Any model diff remains: do not finish. |

Failed predictions remain **REFUTED** in the receipt.  A post-hoc pair is not
eligible to land.  A boundary is called a compensating owner only if the
mechanical ordered predicate fires; smaller error at a later aggregate is not
silently promoted to attribution.

## Required output and OPEN

Commit the gate and controls, run base and the exact round-59 UP3 arm, revert
the arm, run focused tests, independent read-only review, citation gate and a
real shifted-line plant, then the one required ocean-fidelity battery.  The
receipt reports every boundary, including `UNMEASURED` rows, and labels the
claim frame.

If an owner is named, the next round preregisters that one statement as the
pair with source-ordered UP3.  Otherwise continue at the first unmeasured
boundary.  The QCO/RK candidate stays held behind this walk.  Decision 52's
independent ORCA2 start and month-scale ranking follow the step walk.  Sea ice
remains unchanged at `STOP_SELECTOR_GAP`.

ASKED: walk the held UP3 movement from kt=3 stage-2 after-ADV through kt=4 entry.  
UNASKED: configuration, carried-state, stabiliser, and sea-ice changes.
