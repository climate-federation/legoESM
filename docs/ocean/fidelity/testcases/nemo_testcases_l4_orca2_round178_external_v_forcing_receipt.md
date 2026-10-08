# ORCA2 round 178 — independent external V-forcing boundary

Date: 2026-10-08. Base `86c2a9051`; measurement commit `8898b21da`.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round178/`.
Status: **HELD**.

Every trajectory number is **independent hierarchy rung 0**. No
given-NEMO-entry result is mixed into this receipt. No production model,
configuration, carried-state rule, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Verdict

The corrected independent entry is exact on every active T/S/u/v/ssh cell and
the admitted rank-complete round-96 record remains usable. In compiled source
order, completed SSH and U slow forcing are at the floor. The first non-bit
statement is NEMO's direct assignment of the completed V momentum RHS,
`zv_frc(:,:) = Ve_rhs(:,:)`
(`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:287-293`).

The discrepancy is confined to 68 northern-fold cells on row 147: complete-
recorded-domain RMS `5.430611438169775e-08`, maximum
`2.4423127429248864e-06` at `[147,135]`. Those faces are outside the compact
prognostic V mask, so the active-only score is zero, but they feed neighbouring
active continuity stencils. The gate therefore applies round 129's
mask-independent operand rule and stops at V forcing; it does not relabel the
downstream substep-2 SSH result as the owner.

The one-variable completed-forcing arm refuses a landing. Baseline external
SSH endpoint RMS is `0.023847893734247` m; substituting NEMO's completed slow
forcing changes it to `0.024770955736879513` m (away). The maximum remains
`0.3943757874603817` m at `[143,124]`. The history-only arm is bitwise null,
and adding the exact histories to the slow-forcing arm changes nothing. Thus
the compiled assignment names the first boundary, but the first operator that
builds the non-bit `Ve_rhs` remains unmeasured and no statement lands.

## Source-order table

| boundary | comparison domain | unequal cells | RMS | maximum | verdict |
|---|---|---:|---:|---:|---|
| completed SSH forcing | complete record | 0 | 0 | 0 | bit-exact |
| completed U forcing | complete record | 0 | 0 | 0 | bit-exact |
| completed V forcing | complete record | 68 | `5.430611438169775e-08` | `2.4423127429248864e-06` | **first debt** |

The record is parsed from its self-describing headers: both rank shards cover
148x180 exactly once and carry all 65 substeps. The passive trace reproduces
ordinary-solver SSH, U/V, external U/V and transport U/V arrays bit-for-bit.
The walk stops immediately after the first debt.

## Loud correction to round 177

Round 177's broad statement that the independent T entry was bit-exact over
all 799,200 stored cells was wrong: that ladder row had installed the NEMO
frame. The unbridged private card is exact on all 430,552 active T cells, but
151,917 inactive cells differ in the sign bit of zero; numerical maximum is
0.0. S/u/v/ssh are full-storage bit-exact. This does not alter the executable
entry or the downstream independent result, but it retracts the full-storage
wording. The correction is also written into the round-177 receipt, and the
round-178 gate distinguishes active values from stored bits.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R178-P1 record admits unchanged | **CONFIRMED**. |
| R178-P2 active entry and initial histories are exact | **CONFIRMED**. |
| R178-P3 history substitution is null | **CONFIRMED**. |
| R178-P4 completed slow forcing owns the first debt | **CONFIRMED**: V forcing is first. |
| R178-P5 slow forcing improves the endpoint | **REFUTED**: RMS moves away; maximum is unchanged. |
| R178-P6 measurement only | **CONFIRMED**. |

The preregistration's phrase that bit exactness requires `np.array_equal` was
also too weak for signed zero. The committed gate uses uint64 bit comparison;
its signed-zero unit control proves the distinction.

## Mechanical controls, tests, and review

Rank-placement, record-bit, source-order, arm-identity and endpoint-ULP plants
all exit 2 with `STATUS PLANT-FIRED`. Focused round-178 and record-parser tests
pass. The required separate read-only Codex review was attempted and returned
**independent review unavailable in-sandbox**: `failed to initialize
in-process app-server client: Read-only file system`.

Artifact:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round178/external_ssh_walk.json`.

## OPEN

1. Replay the operators that form kt=1 stage-1 `Ve_rhs` offline, in NEMO's
   compiled accumulation order, from the passive completed stage entry. Stop
   at the first operator whose complete 2-D V accumulator differs on the 68
   fold cells; do not add an in-executable observer.
2. Split that operator's operands one variable at a time and name its first
   NEMO-cited statement. Re-test the already-held V-transport/halo unit only
   after this upstream forcing boundary is closed.
3. Then rerun the independent month and attribute the step-96 live-thickness
   refusal only if it survives. The 240-step score remains
   **UNMEASURED-with-spec**.

No acquisition is needed: the admitted RHS/operator and split-explicit records
already contain the required kt=1 operands. ASKED choices are the round-178
source-order walk. UNASKED choices are empty.
