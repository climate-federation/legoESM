# Preregistration — ORCA2 round 153 V transport-mask causal arm

Date: 2026-10-05. Frozen base: `e1b0390b1`.
Claim labels: hierarchy rung 0 is **independent**; the shipped rung-7 card is
**given NEMO's recorded entry**. These labels remain separate in every table.
No configuration, initial-state, forcing, carried-state, stabiliser, sea-ice,
or `unmeasured_features` choice is authorised.

## Source statement and candidate

The compiled rung-0 oracle forms the substep V metric transport as
`zhV = e1v * va_e * zhvp2_e` over its extended V loop, with no face-mask
factor, at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`.
It immediately consumes the north-minus-south transport difference in the
continuity and sea-surface statements at `dynspg_ts.f90:584-591`.

Round 152 measured all three operands and the written unmasked two-product
bit-exact over the complete 26,640-cell record. The current shared helper
instead multiplies the V result by the compact `v_mask`; that extra factor
creates 68 northern-fold-row differences and its offline removal closes the
transport, V difference, and sea-surface replay. This round tests that one
statement through the production JIT before considering any landing.

## Frozen protocol

1. Add one private, default-off test hook which omits only the final compact
   mask multiplication from the literal V metric transport. It is not a
   public card/config field. U transport, operands, source rounding, loop
   association, continuity, forcing, clocks, masks everywhere else, and all
   six sea-ice selectors remain unchanged.
2. Extend the existing round-146 production-JIT gate rather than implementing
   a second solver. Prove the unset hook returns byte-identical ordinary state
   and trace pytrees. With the hook armed, require the registered substep-2
   `transport_v`, `continuity_dv`, and `after_ssh` rows to close through the
   actual model step.
3. If the causal chain closes, run the independent rung-0 and given-entry
   rung-7 kt=1..10 ladders at the frozen base and with the private arm. Compare
   all 200 rows per ladder, register every move and status change, and retain
   the first-non-bit checkpoint predicate.
4. The approximately 31 PSU compensation is a hard veto: at kt=10 stage 3,
   neither ladder's salinity maximum may increase from its same-tree control.
   Any increase is a failed prediction and leaves production unchanged.
5. Only if both ladders pass may the mask removal become production. A landing
   then requires GYRE base/tip ten-step residual equality and 30-day snapshot
   byte identity, plus DINO, LOCK_EXCHANGE, OVERFLOW, generic-card, citation,
   focused-test, and push-gate coverage. Any shared-card regression holds the
   change; no per-card exception is allowed.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R153-P1 | The default-off hook is passive. | Ordinary state and every exposed trace leaf are `np.array_equal` with the hook absent versus explicitly false. | Any default/control bit moves. |
| R153-P2 | Omitting only the V transport mask closes the measured production chain. | Substep-2 `transport_v`, `continuity_dv`, and `after_ssh` each move 68 unequal cells to 0; prerequisites remain exact. | Any target remains non-bit or any earlier exact prerequisite moves. |
| R153-P3 | The candidate does not sacrifice an exact ladder row or advance the first-debt checkpoint. | Both 200-row ladders complete, no exact row leaves exact, and each first non-bit checkpoint is unchanged. | Either ladder refuses, loses an exact row, or moves first debt earlier. |
| R153-P4 | The candidate is salinity-safe. | Each kt=10 stage-3 salinity maximum is no greater than its same-tree control. | Either salinity maximum increases; in particular, a return of the registered approximately 31 PSU exposure is a hard veto. |
| R153-P5 | If production changes, every shared executing card passes its standing gate. | GYRE, DINO, tanks, generic cards, citations, and tests all pass their registered predicates. | Any unregistered or over-floor regression. |

## Controls and terminal rule

The existing round-152 registry and one-ULP plants must still fire. Add a
non-vacuity control that proves the private arm changes exactly the registered
68 V-transport cells, and a ladder comparison control that perturbs one exact
row and makes the comparison refuse. A causal close is not a landing by
itself. If R153-P4 fails, retain the default-off instrument, record the
compensating error, and end **HELD** with production unchanged.

ASKED choices: none. UNASKED choices: empty.
