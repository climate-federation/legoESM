# ORCA2 round 141 — finite-magnitude growth acquisition

Date: 2026-10-04. Base: `770095c16`. Preregistration: `81211759e`.
Acquisition producer: `752759e57`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round141`.
Verdict: **STOPPED_FOR_RECORD**. No ocean model, card, physics, deck physics,
selector, carried state, stabilizer, sea-ice field, or `unmeasured_features`
entry changes in this round.

Every requested trajectory value is labelled **independent**: hierarchy rung
0 starts from its own climatological T/S, zero velocity, and zero sea surface.
No Decision-52 recorded-entry value is mixed into this record.

## Frozen ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R141-P1 | CONFIRMED | The admitted roots contain step-1..10 restarts/frames and a terminal step-240 restart, but no rank-complete step-30..36 stream carrying entry SSH/thickness, external mode, and stage-1 transports. |
| R141-P2 | UNMEASURED | NEMO's step-30..36 values do not exist yet; no first-over-floor step is inferred from legoESM alone. |
| R141-P3 | UNMEASURED | The missing record prevents ordering the sea-surface/barotropic group against the stage-1 transport group. |
| R141-P4 | UNMEASURED | The additions-only source/format preflight passes and its three launcher controls fire; byte-identical kt=1..10 restart proof awaits the actual calibration run. |
| R141-P5 | CONFIRMED | The committed tree changes only the acquisition, checker/test, citation map, preregistration, and receipt. The `packages/` diff is empty. |

## Missing record and compiled order

The exact missing NEMO streams are, on both MPI ranks and at each of steps
30 through 36:

1. step entry `ssh`, `r3t(Kbb)`, `uu_b(Kbb)`, and `vv_b(Kbb)`;
2. split-explicit output `ssh(Kaa)`, `r3t(Kaa)`, `uu_b(Kaa)`, `vv_b(Kaa)`,
   `un_adv`, and `vn_adv`;
3. stage-1 `r3t(Kaa)`, `zFu`, `zFv`, and `zFw`.

This is the source order the rung-0 build executes. The compiled step program
calls the split-explicit solve before stage 1 at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/stprk3.f90:204-215`. The solver starts
from the selected sea surface, barotropic velocities, and `r3u/r3v` depth
ratios at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:359-375`, then finalizes
the depth-mean transports, external velocities, and sea surface at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:890-960`. The stage-1
transport program constructs the horizontal transports at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/traadv.f90:170-235`, then evaluates
`wzv` and forms `zFw` at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/traadv.f90:267-315`.

Round 83's month record cannot answer this question: it contains only the
step-240 terminal restart. The round-91/96 frame and split-explicit records end
at step 10. Restarts expose neither the stage-1 transports nor the precise
source boundary requested by note B48. Reconstructing them from legoESM would
violate the oracle-first rule.

## Acquisition contract

The requested launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round141_growth_acquisition/run.sh`.
It derives from the admitted Decision-83 rung-0 build
`ORCA2_OMIP_L4_R96SPG`, pins its two patched sources, cpp keys, binary,
namelist, and input/deck manifests by content, and creates a fresh target
`ORCA2_OMIP_L4_R141GROWTH`. The source patch is additions-only.

The launcher performs two runs of the same new binary:

- an unchanged 10-step calibration whose 20 restart shards must be
  byte-identical to the admitted round-96 shards;
- a 36-step independent run whose only namelist delta is the run protocol
  (`nn_itend=36`, terminal stock at 36, no restart list).

Each rank writes one self-describing stream per step 30..36. Every file carries
magic, version, step, rank, local/global layout, precision, field count, and a
per-field `(name, rank, n1, n2, n3, payload)` description. The checker derives
all payload lengths from those headers, requires exactly-once global coverage
at every step, requires all fields finite, and refuses any restart movement.
Its header, field-name, field-rank, field-dimension, truncation, missing-step,
nonfinite, swapped-rank, and restart-byte plants must all fire during admission.

The committed preflight passes source application plus Fortran syntax for the
writer, `stprk3`, and `traadv`. The source-layout, hidden-deck-delta, and
producer-content launcher plants each emit `STATUS PLANT-FIRED`. The checker
unit battery passes 11/11, including all nine binary-record plants.

## Gates, tests, and review

There is no `packages/` change, so no ORCA2, GYRE, DINO, tank, or generic-card
trajectory can move in this record-only round. The production model tree is
identical to base `770095c16`; the acquisition's kt=10 restart identity is a
hard admission predicate, not an argument from an unchanged diff.

The focused round-141 battery passes 11/11. Citation gates and the required
full fidelity battery are run on the final receipt commit; their outputs are
stored under the round-141 evidence root. The separate `codex exec --sandbox
read-only` review could not initialize with `failed to initialize in-process
app-server client: Read-only file system`. Verdict: **independent review
unavailable in-sandbox**.

## OPEN

1. The operator runs the committed launcher. Admission must produce
   `PASS_R141_GROWTH_RECORD`; otherwise every science value is rejected.
2. Once admitted, print NEMO versus legoESM at the target column and incident
   faces for steps 30..36 in the recorded source order. The first step and row
   beyond `2e-10` own the next one-variable walk.
3. After a cited statement closes that boundary under the standing gates,
   rerun the independent rung-0 month and report its new first non-finite step
   or the complete 240-step score. Decision 94 remains pending and untouched.
