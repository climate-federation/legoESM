# Preregistration — ORCA2 round 176 independent stage-1 offline replay

Date: 2026-10-08. Frozen base: `8f3aaa0aa`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round176/`.

Every scientific number is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No rung-7
given-NEMO-entry number is mixed into this round. Sea ice, all six sea-ice
selectors and the shipped card's `unmeasured_features` tuple remain unchanged.

## Frozen record, cells and source order

The oracle is the admitted round-175 rank-complete record. Its two
self-describing files carry 23 fields at kt=1 stage 1 and cover the 148x180
global domain exactly once. The additions-only admission reports all 20 ocean
restart shards byte-identical to the round-90 producer and status
`PASS_R175_STAGE1_ADMISSION`.

Two cells are scored separately and may not be conflated:

1. `[j=147,i=49,k=0]`, the machine-selected stage-1 salinity maximum from
   round 174 (`3.2847473521544472 PSU`);
2. `[j=86,i=159,k=3]`, the production rung-0 month's registered failure
   column, whose stage-1 value was not measured by round 174.

The compiled rung-0 source order is the external-mode/QCO handoff, completed
momentum RHS, velocity update, barotropic correction, metric transports,
centred tracer advection, surface tracer source, QCO tracer update, and final
boundary association
(`ORCA2_OMIP_L4_R175STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:126-259`,
`:449-601`, `:621-717`, `:773-898`). The offline replay calls the existing
pure literal helpers on NEMO's recorded operands. It never adds a callback,
returned diagnostic, observer or hook to the executable model.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R176-P1 | The operator-run round-175 record remains admissible without reinterpretation. | Both rank files parse from their own headers, cover the global domain exactly once, all 23 fields are finite on owned cells, and the 20 restart comparisons remain byte-identical. | Any header, field, placement, coverage, finiteness or restart comparison fails. |
| R176-P2 | Both registered cells are present and are reported separately. | Every source row reports each cell independently; the stage maximum remains `[147,49,0]`, while `[86,159,3]` gets its own measured value. | Either cell is absent, their values are merged, or prose location replaces the machine argmax. |
| R176-P3 | Given NEMO's own recorded inputs, the existing literal stage-1 momentum update, barotropic correction, metric-transport and centred-advection helpers stay at the `2e-10` floor at both cells and globally on their active domains. | Every replay row through centred tracer advection is at or below the floor; bit-exactness is reported separately and is never inferred from tolerance. | The first source-ordered replay row above the floor names that operator and terminates the walk below it. |
| R176-P4 | Rung 0 has no stage-1 surface tracer source, so NEMO's after-advection and after-SBC accumulators are bit-identical. | `adv_t == sbc_t` and `adv_s == sbc_s` bit-for-bit on every owned active cell. | Either accumulator moves. |
| R176-P5 | If every replayed statement is at the floor, the 3.2847-PSU completed-stage error is carried in through an upstream candidate operand, not created by the replayed stage statements. | All oracle-input replays stay at the floor while the complete-arm completed stage retains its registered debt; status is HELD at the first candidate operand not yet reproducible offline. | A replayed statement itself first leaves the floor. |
| R176-P6 | This round is measurement-only. | No `packages/`, card, configuration, carried state, stabiliser, threshold, sea-ice selector or production halo path changes. | Any such change lands. |

Failed predictions remain in the receipt. The quantity floor is the frozen
`2e-10`; exact means `np.array_equal` on binary64 arrays. An at-floor row that
is not exact is labelled `AT_BAR_NOT_EXACT`, never discharged.

## Controls and terminal rule

The gate must reject plants in rank placement, field order, source order,
cell identity, active mask, first-debt selection and a one-ULP output. The
record parser derives payload sizes from each field header. Candidate, oracle
and geometry arrays must be float64; execution is CPU with fp64/libm policy.

The first replayed operator above the floor owns the next operand walk. If no
replayed operator crosses the floor, stop at the first upstream candidate
operand that is not available without observing the executable; do not infer
an owner and do not add an observer. No model statement lands in this round.

ASKED choices: continue the independent rung-0 kt=1 stage-1 walk using the
admitted rank-complete record and offline pure-function replay.  
UNASKED choices: empty.
