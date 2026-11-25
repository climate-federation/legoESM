# ORCA2 round 269 — G1 transfer Decision-96 hold

Date: 2026-10-13  
Lane base: `770798365e09e90e82823ad0846aac5a8dc336e3`  
Labels: **independent** and **given NEMO's entry**, reported separately  
Verdict: **HELD**. The imported G1 statements were retired; no model or card
change lands.

## 1. Frozen unit and compiled statements

The preregistration is
`PREREG_nemo_testcases_l4_orca2_round269.md` at `45a9718c6`. The exact package
implementations came from pinned geometry-ladder tip
`b0fa6a3f5cc2b957900dacd1a72fcbbd70552d9c`, not from a re-derived scheme.
The operator called the transfer five statements; mechanically it contains
five landing groups and six compiled statements:

1. EEN builds live U/V transports and consumes the fold-image V row in its
   triads at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:784-785` and
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynvor.f90:790-802`;
2. the split-explicit midpoint U/V depths use the adjacent T-point SSH values
   at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:509-520`;
3. the hybrid stage-1 sea level is the written two-product sum at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:138-178`;
4. the external substep is a true quotient at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:1077`;
5. the flux-form velocity bracket is multiplied by the mask-safe inverse exit
   depth at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:655-718`;
6. its loop-entry face depth is reference depth times one plus the live ratio
   at
   `ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:361-364`.

The execution census was explicit. OMT-0 selected the two flux-form-only
statements. OMT-1..4, rung 0 and rung 10 selected the four shared/vector
statements and explicitly left the two flux-only selectors off. The six
sea-ice selectors, every deck, initial/carried state and the ORCA2 card's
`unmeasured_features` tuple were unchanged.

Candidate commits were `8b0ed957f` and `526346399`; the already-certified
matched-stage observer was restored at `0307cd5b7`. After the frozen landing
predicate failed, the observer and candidate were retracted in
`a9cd8d07e`, `e5c57817b` and `72b213630`. At the receipt tip,
`git diff 770798365 -- packages tests/ocean/unit/test_orca2_g1_transfer.py
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round204_omt0_ladder_gate.py
scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round209_omt1_ladder_gate.py`
is empty.

## 2. Matched stage-2 statement

The passive source-order replay passed record identity, completed-state
passivity, accumulator self-replay and the one-bit plant under both labels.
The pre-VOR accumulator is below the `2e-10 m s-2` floor; the first valued
boundary remains the stage-2 EEN vorticity accumulator.

| label | after-VOR max before | after-VOR max selected | RMS selected | argmax |
|---|---:|---:|---:|---|
| independent | 1.7082229745349855e-6 | 1.2863307536456779e-8 | 1.2191816847794566e-10 | `[19,33,3]` |
| given NEMO's entry | 1.7082229745349855e-6 | 1.2863307536456779e-8 | 1.2191816847794566e-10 | `[19,33,3]` |

Thus R269-P3 is **CONFIRMED**: the fold-image statements preserve round
268's approximately 133-fold improvement. They do not close the interior EEN
residual. The accumulator addition self-replays to
`3.3881317890172014e-21 m s-2`, so the residual is an EEN operand, not an
observer or accumulation artefact.

## 3. Decision-96 card census

Every comparison used the same current tree and admitted record. The
selector-off control disabled all six G1 selectors on that tree, avoiding a
comparison against a historically different implementation. `moved` includes
bit- or score-moved rows; `equal` below means RMS score-equal and therefore is
not a vote.

| card / label | rows | moved | RMS toward | RMS away | RMS equal | exact rows lost | final stage-3 SSH max before -> selected |
|---|---:|---:|---:|---:|---:|---:|---:|
| OMT-1 independent | 160 | 140 | 133 | 7 | 0 | 0 | 0.7561238239857295 -> 0.746821544968785 |
| OMT-1 given NEMO's entry | 160 | 140 | 133 | 7 | 0 | 0 | 0.7561238239857295 -> 0.746821544968785 |
| OMT-2 independent | 200 | 180 | 165 | 15 | 0 | 0 | 0.38416224105021546 -> 0.3464894384404165 |
| OMT-2 given NEMO's entry | 200 | 180 | 165 | 15 | 0 | 0 | 0.38416224105021546 -> 0.3464894384404165 |
| OMT-3 independent | 200 | 180 | 162 | 18 | 0 | 0 | 0.36543082189181 -> 0.36353996790232357 |
| OMT-3 given NEMO's entry | 200 | 180 | 162 | 18 | 0 | 0 | 0.36543082189181 -> 0.36353996790232357 |
| OMT-4 independent | 200 | 184 | 160 | 24 | 0 | 0 | 0.3948305183668017 -> 0.39374085285266497 |
| OMT-4 given NEMO's entry | 200 | 184 | 160 | 24 | 0 | 0 | 0.3948305183668017 -> 0.39374085285266497 |
| rung 0 independent | 200 | 200 | 159 | 25 | 16 | 0 | **0.3840334615160934 -> 0.3840334668286476** |

All four OMT cards pass their frozen Decision-96 predicates under both
labels. Rung 0 has a large RMS majority toward NEMO and loses no exact row,
but its sea-surface maximum worsens by `5.3125542e-9 m`. The first over-bar
row, kt=1 stage-1 T, is unchanged (`RMS 1.1745202172532108e-5 K`; maximum
`1.3731656162621064e-3 K`). The rung-0 classifier therefore refuses exactly
at `final stage-3 SSH maximum worsened`.

R269-P2 is **REFUTED**, and the preregistration says any such failure holds
the transfer. Rung-10, GYRE, DINO and tank runs became off-claim once this
red was known because the final tree carries no package change. No result for
those gates is claimed or implied.

## 4. Controls and evidence

`nemo_testcase_l4_orca2_round269_g1_decision96_gate.py` enforces a clean
candidate commit, current-tree selector-off control, complete row keys, no
exact-row loss, RMS majority, unchanged/toward first row and non-worsening
final SSH maximum. Its planted exact-row loss exits 2. The inherited matched
stage gate's one-bit plant also exits 2.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round269/`.
SHA-256 prefixes: independent/given after-VOR `00a7617eb847552b` /
`c75649508ee32341`; after-VOR classification `0edef649f71db453`; OMT-1
Decision-96 `b807432cd697b1a5` / `ae82de309c637aa7`; OMT-2
`48ed39dee1bada46` / `7305e38246fbc9bc`; OMT-3
`455f4f9488b03f19` / `df34ff99102be12b`; OMT-4
`502a30e4e3227c34` / `08a4b5f7de3d39a6`; rung-0 candidate/control
`a22ef3212ab4183b` / `937fbd48659d699f`; exact-loss plant
`ac07eba8845440ef`; one-bit plant `ac75c5187f692789`.

## 5. Validation, review and citation audit

Focused transfer tests on the candidate: **5 passed**. On the final tree the
receipt-citation tests plus the five GYRE recipe regressions that caught the
round-241 shared-path defect pass **22/22** serially in `488.91 s`.

The required `tests/ocean/fidelity -n 12` battery was run once. It reached
**98% with no reported failure**, after which every pytest worker had exited
but the parent `tee`/waiter emitted no summary; the orphaned waiter was
interrupted (exit 130). This is the campaign's documented xdist tail-hang
class, not a test failure, and the one-battery rule forbids a parallel or full
rerun. The complete 6,840-line log is
`round269/fidelity_battery.log` (SHA-256
`0285fb94a150c526ba9cc6088c5aed81138bf569ff2caea7e6fdc94d1f5d1ddb`).

The round receipt citation gate passes **7/7** citations with zero failures,
zero unmapped citations and a clean map audit. Its shifted line-1077
`dynspg_ts.f90` plant fires. The cumulative default receipt passes
**274/274** with the same zero-failure/zero-unmapped result.

The required separate `codex exec --sandbox read-only` review was attempted,
but the client failed before reading the diff with `failed to initialize
in-process app-server client: Read-only file system`; therefore **independent
review unavailable in-sandbox**. The three-line failure log is
`round269/independent_review.log` (SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).

## OPEN

1. On rung 0 only, split the held shared statements into the preregistered
   source units: EEN fold-image row; midpoint face depth; stage-1 SSH; true
   quotient. The OMT-0-only flux pair cannot own the rung-0 red because rung 0
   never selects it. Measure the kt=10 stage-3 SSH maximum after each unit;
   do not land a partial unit without its own frozen Decision-96 census.
2. If one or more cited units pass independently, run rung 10, GYRE, DINO,
   tanks and the full moved-row gates for only those units before landing.
3. The EEN leaf replay remains after a successful fold-image-row landing:
   `ff_f`; relative circulation; `r1_e1e2f`; live
   `e3f_0vor*(1+r3f)`; `zwx/zwy`; triads; final metric/addition. R269-P5 was
   **NOT REACHED** because its precondition was “after landing.”

No NEMO acquisition and no user decision are needed for the next measurement
round.
