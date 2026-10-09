# ORCA2 round 203 preregistration — recover the OMT-0 ten-step record

Date: 2026-10-09. Frozen base: `4874e52ee0`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round203/`.
All state comparisons are labelled **independent OMT-0**. The shipped rung-10
ORCA2 card, sea ice, its six selectors and `unmeasured_features` are unchanged.

## Prior evidence and scope

The operator's round-202 frequency run is prior evidence. It created only the
rank-complete step-1 restart and then NEMO stopped before integration because
`nn_stock=1` is not divisible by the unchanged surface cadence `nn_fsbc=2`.
Compiled `sbcmod.f90:346-349` owns that refusal. Compiled
`restart.f90:94-119` proves that supported list-mode targets must be opened one
step before writing, and `restart.f90:188-202` proves that a just-written list
target is closed before the next target is selected. Therefore adjacent list
targets cannot provide the requested record.

This round changes only the record protocol. It keeps the exact OMT-0 deck,
binary, inputs and `nn_fsbc=2`. Two deterministic composite twins are built
from independent NEMO runs: an expected cadence-refusal run supplies step 1;
supported non-adjacent list runs supply even steps `(2,4,6,8,10)` and odd steps
`(3,5,7,9)`. The already measured kt=11 month boundary remains the month
record. No NEMO source patch, physics selector, stabiliser or threshold change
is permitted.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R203-P1 | Round 202 failed only on the compiled cadence guard. | Its resolved deck has `nn_stock=1`, `nn_fsbc=2`; both rank shards carry finite fp64 `kt=1`; no step-2 shard exists; the output contains the exact `sbc_init` refusal. | Wrong `kt`, field, dtype, non-finite payload, later shard or another NEMO error: **REFUTED**; do not reuse it. |
| R203-P2 | A second identical cadence-refusal run reproduces the step-1 entry bit-for-bit. | Both ranks and all five fields are array-equal to round 202; the run refuses only through the same cadence guard. | Any unequal field or different stop: **REFUTED**; no composite record. |
| R203-P3 | Four supported list-mode runs complete the two twins. | Even-A/B write exactly steps 2,4,6,8,10; odd-A/B write exactly 3,5,7,9; every run reaches `STOP 0`; all 36 shards are finite fp64 and carry their filename's `kt`. | Missing/wrong shard, unexpected restart, non-finite payload or nonzero completion: **REFUTED**. |
| R203-P4 | The composite twins are deterministic and calibrated. | Steps 1..10 provide 40 rank-complete shards and 100 array-equal twin field comparisons; both even step-10 states equal the admitted OMT-0 month step-10 state in 20 comparisons. | Any unequal field: **REFUTED**; no card score. |
| R203-P5 | The record gate and launcher are non-vacuous. | Plants for cadence, adjacency, missing component/rank, wrong step, non-finite payload, twin ULP, calibration ULP, unexpected restart and changed binary all refuse. | Any green/no-op plant: instrument invalid; cite no record number. |
| R203-P6 | Recovery changes no model or card behavior. | `git diff 4874e52ee0..HEAD -- packages` is empty and the resolved physical namelist diff remains exactly Decision 103's five OFF modules. | Any package or extra physical deck delta: stop and remove it. |

## Landing predicate

The round may admit the recovered record and land measurement machinery only
after R203-P1..P6 pass. The OMT-0 card and trajectory are still blocked until
the operator supplies and the gate admits every component. OMT-1 does not
begin. Every failed prediction remains recorded as **REFUTED**.

ASKED choices: Decision 103's OMT-0 deck and ten-step record.
UNASKED choices: empty.

## Frozen addendum after R203-P1 refutation

R203-P1 is **REFUTED and retained**: both cadence-refused files carry `kt=1`
but none of the five ocean fields. They are open NetCDF headers, not state
records. No value from them will be used.

The replacement acquisition is frozen before execution. It reuses the
round-90 additions-only frame writer already admitted on the same rung-0 deck:
compiled `stprk3.f90:103-103,223-240` writes the `Nbb` step entry and three
completed RK stages, while compiled `restart.f90:176-180` independently names
the restart payload as the before fields. Two instrumented OMT-0 runs must each
produce 80 self-describing frames (two ranks, kt=1..10, stages 0..3, five fp64
fields). A third uninstrumented OMT-0 run uses the supported single target
`(10,)`; both instrumented terminal restarts must be byte-identical to it.
The two frame records must compare array-equal in 400 field comparisons.

This addendum supersedes only R203-P2..P4's staggered-list mechanism, not their
determinism, coverage or calibration predicates. The cadence, malformed-header,
wrong-field, truncation, non-finite, missing-frame, twin-ULP, terminal-restart
and changed-binary plants must all fire. Any writer-induced terminal bit change
or frame twin difference stops the round.
