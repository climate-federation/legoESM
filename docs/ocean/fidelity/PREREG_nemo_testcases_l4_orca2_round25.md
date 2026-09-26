# NEMO testcase Lane 4 — ORCA2 card round 25 preregistration

Date: 2026-09-26

Parent: `cae4be7af0c7cf9becbd6f6bac87cdb560af5e2c`

Status: **PREREGISTERED BEFORE ROUND-25 SCIENTIFIC SCORING.**

Round 25 executes round 24's OPEN item: substitute the three authorized
Decision-54 statements separately on the independent ORCA2 trajectory before
any landing retry.  The three clean, one-variable arms are:

1. `single_mask`: the file-read `ahmf` is not multiplied by a second binary
   vertex mask;
2. `live_thickness`: the LDF operator receives NEMO's recorded T/U/V/F mesh
   and its live Kbb/Kmm thicknesses, while retaining the parent metric path;
3. `native_f_metrics`: the LDF vorticity circulation receives NEMO's stored
   `e1f*e2f` reciprocal and native edge reciprocals, while retaining the
   parent mask and thickness paths.

Every arm starts from the clean parent and is reverted before the next arm.
Every ORCA2 ladder number is labelled **independent with Decision-52 SSH**.
No given-NEMO-entry operator score is mixed into a trajectory table.  The six
sea-ice selectors and the card's `unmeasured_features` tuple remain frozen.

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round25/`.

## Compiled statements and executed branch

The executing build reads `ahmt_3d` and `ahmf_3d` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/ldfdyn.f90:348-353`, then
applies their masks exactly once at `:387-393`.  WS-RK3 calls `dyn_ldf` at
stage 3 with Kbb/Kmm operands at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stprk3_stg.f90:493`.
The executing level operator uses the already-masked F coefficient, live F
thickness, stored F-cell area reciprocal, circulation edges, live T/U/V
thicknesses, and Kmm face divisors at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/dynldf_lev.f90:123-140`.

Each arm uses the same card, admitted NEMO record, surface forcing, CPU
backend, fp64 policy, ten-step maximum, and Decision-52 sea-surface entry.
No arm changes a resolved configuration leaf.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R25-P1 | Each arm is a one-variable execution of exactly one registered Decision-54 input boundary. | The clean arm diff contains only that input boundary, the resolved card is unchanged, and at least one ladder row moves. | An arm changes another boundary/config leaf, or no row moves. |
| R25-P2 | Each live arm first moves at kt=1 stage-2 momentum while leaving the first NEMO mismatch at kt=1 stage-1 T. | Entry and stage 1 equal the parent; the first moved row is stage-2 u or v; the first non-bit statement and score are unchanged. | An earlier row moves, no stage-2 momentum row moves, or the first NEMO mismatch changes. |
| R25-P3 | The live-thickness arm is the component that independently exposes round 24's non-positive raw-`e3w` refusal. | It completes kt=1..3 and refuses while entering kt=4 with the registered message. | It reaches kt=4, or refuses earlier/differently. |
| R25-P4 | The single-mask and native-F-metric arms remain finite through kt=10. | Each produces 40 checkpoints and `LADDER_MEASURED`. | Either refuses before kt=10 or produces a non-finite state. |
| R25-P5 | No arm moves a formerly bit-identical row off the bar or moves the first-over-bar boundary earlier. | The outcome gate reports an empty AT-BAR-left set and the same first non-bit statement for every arm. | Any formerly exact row becomes non-bit or the owner boundary moves earlier. |
| R25-P6 | The comparison instrument can fail. | A one-ULP plant in one otherwise unchanged arm row is refused, and the citation gate's rigid shift fires. | Either planted violation passes. |

Failed predictions remain **REFUTED**.  If more than one single arm exposes
the refusal, all such arms are named; no post-hoc winner is selected.  If no
single arm exposes it, the measured pair interaction becomes the OPEN item
and Decision 54 remains held.

## Landing and stop rules

- Commit this preregistration before running any arm.
- Run each arm from the same clean parent and retain its clean-worktree stamp.
- Register every moved field row and its direction against the admitted
  round-23 ten-step baseline; never synthesize rows beyond a refusal.
- Do not relax the raw-`e3w` refusal or add a stabilizer.
- This is an attribution round.  No model statement lands, even if one arm is
  benign; all experimental model changes are reverted before the receipt.
- No NEMO acquisition, configuration choice, carried-state change, sea-ice
  edit, or NEMO source edit is authorized.

## Choices

ASKED: Decision 54 and round 24's OPEN item authorize separate measurement of
the three already-cited inputs.

UNASKED: none.
