# ORCA2 round 175 — independent kt=1 stage-1 record boundary

**Status:** `HELD / ACQUISITION_NEEDED`  
**Claim label:** every scientific number is **independent**: hierarchy rung 0
starts from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into this receipt.  
**Evidence:**
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round175/`.

## Verdict

The walk stops at its first unavailable oracle boundary. The admitted legacy
stage-1 tracer stream is an unranked rank-0 file whose owned slab contains
global longitudes 0--89. The requested production-failure column at global
`[j=86,i=159,k=3]` is therefore absent. No internal operator and no NEMO
statement is named this round.

Two frozen premises are **REFUTED**, not hidden:

1. R175-P1 predicted a shallow four-level partial column. The machine census
   instead finds `mbathy=21`, bottom wet level 20, with levels 0--5 all wet.
2. R175-P3 repeated round 174's prose claim that the stage-1 salinity maximum
   was at `[86,159,3]`. The pinned round-174 JSON places its global
   `3.2847473521544472 PSU` maximum at `[147,49,0]`. The prose location was
   wrong. The 3.2847-PSU number is not a measured value at `[86,159,3]`.

The requested column remains relevant because it is the production rung-0
month's registered failure column. The acquisition is deliberately global and
rank-complete, so the next round can score both the true stage-1 maximum and
the production-failure column without choosing between them post hoc.

## Target-column geometry

NEMO's `mesh_mask` calls its bottom index `mbathy`; the input domain file calls
the same wet-cell quantity `bottom_level`. They are equal on every wet column.
On 10,207 dry columns the output uses level 1 where the input uses 0, a
recorded land convention rather than a wet-cell discrepancy.

| Quantity at `[86,159]` | Value |
|---|---|
| NEMO `mbathy` / zero-based bottom | 21 / 20 |
| partial bottom | yes |
| fold row / cyclic seam | no / no |
| surface W/E/S/N T masks | 0 / 1 / 0 / 1 |
| land-adjacent | yes |
| T/U/V masks, levels 0--5 | all 1 |

| k | NEMO `e3t_0` [m] | NEMO `e3w_0` [m] |
|---:|---:|---:|
| 0 | 10.000015488051758 | 9.999875598728977 |
| 1 | 10.000818315027573 | 10.00035061113249 |
| 2 | 10.002382004484389 | 10.001471046850384 |
| 3 | 10.00542762603709 | 10.003653342421785 |
| 4 | 10.011359574398284 | 10.007903816079306 |
| 5 | 10.022913005705504 | 10.016182389364985 |

At this column legoESM's listed `e3t/e3w` values and T mask are identical to
NEMO. Over the full mesh, `e3t_0`, T mask and surface U mask are bit-exact.
The census also records two existing, unlanded global associations rather than
silently treating them as exact: `e3w_0` differs on 27,326 of 799,200 cells
(maximum 478.67098726506856 m), and the compact surface V mask differs on the
known 68 northern-fold cells.

## Compiled order and first unavailable row

The rung-0 compiled program first saves the external-mode fields and builds
the QCO stretch (`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:137-179`).
It then performs the stage-1 velocity update
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:469-479`) and the
barotropic correction
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:513-537`). The two
resolved transport call sites are
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:555,578`; tracer
advection and the surface source follow at
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:637,645`, followed by
the QCO tracer update
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:678-680` and the
U/V/T/S boundary association
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:791-796`.

| Registered row | Result |
|---|---|
| exact passive step entry | AT-BAR / bit-exact |
| external-mode and QCO operands | UNMEASURED — rank 1 absent |
| completed momentum RHS | UNMEASURED — rank 1 absent |
| momentum before correction | UNMEASURED — rank 1 absent |
| corrected velocity | UNMEASURED — rank 1 absent |
| metric transports | UNMEASURED — rank 1 absent |
| tracer RHS after advection | UNMEASURED — rank 1 absent |
| tracer RHS after surface source | UNMEASURED — rank 1 absent |
| tracer QCO update | UNMEASURED — rank 1 absent |
| completed stage | DEBT globally; max 3.2847473521544472 PSU at `[147,49,0]` |

Selection terminates at the external-mode/QCO row. A value below that absent
row is not used to infer an owner.

## Acquisition and controls

The operator-run launcher is
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round175_stage1_acquisition/run.sh`.
It creates the new target `ORCA2_OMIP_L4_R175STAGE1` and writes one
self-describing file per rank at kt=1 stage 1. Each file declares and carries
23 named fields: external SSH/QCO/barotropic fields, momentum RHS, before/after
barotropic correction, three transports, tracer RHS after advection and the
surface source, QCO-updated T/S, and boundary-associated U/V/T/S.

The checker reads magic, header integers and every field's own name, rank and
dimensions before deriving payload lengths. It requires exactly-once global
coverage and compares all 20 ocean restart shards byte-for-byte with the
admitted round-90 producer, explicitly retaining the two kt=10 terminal
comparisons. Header, field-name, field-dimension, truncation, rank, coverage
and restart-byte plants are wired into admission. They cannot be executed
until the record exists. The local layout plant fires, and the committed patch
passes the Fortran syntax preflight:

`ORCA2_ROUND175_STAGE1_PREFLIGHT_READY .../orca2_rung0_stage1_ranked_10step_np2`.

The record-census gate's target, neighbour classification, rank placement,
boundary order, first-unmeasured selection and one-ULP controls all fire with
exit 2. Its final status is
`STOP_RECORD_R175_STAGE1_RANK_COMPLETE_NEEDED`.

## Prediction ledger

| Prediction | Verdict | Evidence |
|---|---|---|
| R175-P1 shallow four-level target | **REFUTED** | bottom level is 21, not 4 |
| R175-P2 old stream lacks the target rank | **CONFIRMED** | rank 0 only; global i=0--89 |
| R175-P3 pinned boundaries preserve the stated target argmax | **REFUTED** | machine argmax is `[147,49,0]` |
| R175-P4 rank-complete acquisition required | **CONFIRMED** | first internal row absent |
| R175-P5 measurement/acquisition only | **CONFIRMED** | no `packages/`, card, selector or production-path change |

## Review and verification

The requested separate review verdict is **independent review unavailable
in-sandbox**: `codex exec --sandbox read-only` exited before reading the diff
because its in-process app-server client could not initialize on the read-only
filesystem. This is recorded in `independent_review.log`, not represented as a
passing review.

The citation gate passes this receipt and its shift plant fires. The focused
round-175 tests and the full ocean-fidelity battery are reported in the final
committed verification section below. No model file changed, so the ORCA2
ladders, GYRE year, DINO and tanks cannot move by construction and were not
rerun as trajectory claims.

## OPEN

1. The operator runs the acquisition launcher. Admission must pass all seven
   record plants and prove every restart shard byte-identical to round 90.
2. The next round replays source order from the rank-complete record, reporting
   `[147,49,0]` and `[86,159,3]` separately. It names no operator below a
   missing or non-bit boundary.
3. If one cited statement owns both the true stage-1 maximum and the production
   month column, remeasure the month and only then consider the complete private
   arm for atomic landing under the standing gates.
