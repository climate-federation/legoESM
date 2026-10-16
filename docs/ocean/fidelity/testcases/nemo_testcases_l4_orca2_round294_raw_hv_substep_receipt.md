# ORCA2 round 294 — raw-hv substep continuation

Date: 2026-10-16

Status: **HELD**.  On the independent OMT-4 kt=1 stage-1 external solve,
carrying only NEMO's stored V reference depth makes the substep-1 midpoint
V depth bit-exact, confirming round 293.  The next valued source-ordered row
is the same substep's vector-form V velocity update: 35/8,589 active faces,
maximum 0.00027511187960820976 m/s at `[j=147,i=54]`; all 8,489 interior
faces are bit-exact.  The preregistered substep-3 depth prediction is
**REFUTED**.  No package, card, configuration, carried state, or NEMO record
changes in this round.

Every scientific number below is **independent OMT-4, kt=1 stage 1**.  No
given-entry result is mixed into the table.

## Provenance and instrument

- Lane base: `981a93342`.
- Frozen preregistration:
  `PREREG_nemo_testcases_l4_orca2_round294.md`, commit `d2fccd7e2`.
- Committed measurement gate: `64aacec08`; exact V-update citation correction:
  `7f8083aa4`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round294/`.
- NEMO streams: admitted round-222 `oracle_bt_substeps` and
  `oracle_bt_advmean_operands` twins; stage admission: round 280; parent
  closure: round 293's immutable `measurement.json`.
- Execution: CPU, production JIT, fp64/x64, libm.  The observed and
  unobserved raw-`hv_0` arm states are array-identical in every carried field.
- The arm replaces only the compact V reference-depth slot.  Its U slot is
  bit-identical to production (0/26,788 unequal); its V slot is bit-exact
  against NEMO (0/13,320) and differs from production on exactly the inherited
  13 fold-band cells, maximum 719 m, with 0/13,050 interior differences.
- Search-before-build reused round 292's passive source registry and stream
  readers, round 293's causal closure, and the existing reference-depth test
  hook.  No solver, parser, formula, or executable observer was duplicated.
- Two preflight refusals emitted no scientific values and are retained:
  the first supplied a short rather than full producer hash; the second used
  the unhydrated hierarchy deck rather than its input-complete run directory.
  The accepted third execution changed only those invocation arguments.

## Source-ordered result

The record build writes the midpoint V depth from its stored `hv_0` at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:509-520`.
The one-variable arm closes that row at substep 1:

| row | unequal / active support | maximum | interior unequal |
|---|---:|---:|---:|
| production reconstructed `hv_0` vs raw `hv_0` | 13 / 13,320 slab cells | 719 m | 0 / 13,050 |
| raw-`hv_0` substep-1 midpoint V depth vs NEMO | 0 / 8,589 | 0 m | 0 / 8,489 |

Every predecessor in substep 1—slow V, entry V, midpoint V, and midpoint
SSH—is at the campaign floor.  The next valued row is NEMO's vector-form V
update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/dynspg_ts.f90:666-678`:

| region | unequal / support | maximum (m/s) | RMS (m/s) |
|---|---:|---:|---:|
| complete rank slab | 35 / 8,589 | 0.00027511187960820976 | 9.660386936122658e-06 |
| fold band | 35 / 100 | 0.00027511187960820976 | 8.952943078479437e-05 |
| interior | 0 / 8,489 | 0 | 0 |

The row is not attributed below the written update in this round.  The
round-245 surface-mask/T-neighbour depth pair is not credited because it is
downstream of the newly measured first boundary.  Raw `hv_0` remains private
inside the complete fold unit; a direct depth closure is not a trajectory
landing predicate.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R294-P1 | **CONFIRMED** | Records admit; selectors, CPU/fp64/libm, and whole-state passivity are pinned. |
| R294-P2 | **CONFIRMED** | U slot 0 unequal; V slot 13 fold-only differences and 0 unequal against raw NEMO. |
| R294-P3 | **CONFIRMED** | Substep-1 midpoint V depth and every predecessor are at the floor. |
| R294-P4 | **REFUTED** | First valued row is substep-1 `v_exit`, 35 faces, not substep-3 V depth. |
| R294-P5 | **NOT CREDITED** | Round 245's pair is not the first remaining boundary. |
| R294-P6 | **CONFIRMED** | All seven plants print `STATUS PLANT-FIRED` and exit 2. |

## Validation

- Accepted measurement and classification are `measurement3.json` and
  `classification2.json`; their SHA256 values are recorded in the evidence
  `SHA256SUMS` file.
- The round-294 direct known-answer controls pass, including retention of a
  refuted prediction and one-variable arm isolation.
- Citation, focused-suite, full fidelity-battery, and independent-review
  dispositions are recorded below after their gates run; no claim relies on
  an unfinished gate.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's completed-stage top-V exchange, the resolved stage mask,
  raw `hv_0`, and the complete vector-form fold unit remain off in production.
- Decision 126 remains pending; shipped runoff is unchanged and the decision
  is not re-raised.
- Sea ice and the shipped card's `unmeasured_features` tuple are unchanged.

## OPEN

1. Keep raw `hv_0` inside the private fold unit.  Split the substep-1
   vector-form V update at the matched pre-association point in compiled
   order—entry `vn_e`, surface-pressure term `zv_spg`, Coriolis/drag trend
   `zv_trd`, slow forcing `zv_frc`, `rDt_e`, and the surface `ssvmask`—and
   name the first non-bit input on the 35 fold faces.
2. Only after that input is source-exact, re-score the complete held unit
   atomically; no partial face-depth or copied-row landing is allowed.
3. Keep the resolved stage mask and completed-stage top-V exchange held, and
   leave shipped runoff unchanged while Decision 126 remains pending.
