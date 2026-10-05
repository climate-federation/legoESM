# Preregistration — ORCA2 round 154 V transport materialization

Date: 2026-10-05. Frozen base: `459ef0fbb`.
Claim labels: hierarchy rung 0 is **independent**; the shipped rung-7 card is
**given NEMO's recorded entry**. These labels remain separate in every table.
No configuration, initial-state, forcing, carried-state, stabiliser, sea-ice,
or `unmeasured_features` choice is authorised.

## Source statement and candidate

The compiled rung-0 oracle completes and stores the V metric transport as
`zhV = e1v * va_e * zhvp2_e` at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`.
Only afterward does the separate continuity loop evaluate
`zhV(ji,jj) - zhV(ji,jj-1)` and consume it in `zhdiv` and `ssha_e` at
`dynspg_ts.f90:584-591`.

Round 153 proved, through the production CPU JIT and with NEMO's raw reference
V depth held fixed, that omitting legoESM's extra compact V mask makes the
completed metric transport bit-exact over all 26,640 record cells. The next
live north-minus-south statement nevertheless differs at 8,786 cells because
the current helper may fuse the completed product into the subtraction.

This round tests one source-order boundary only: apply the already-shared
`nemo_source_round` IEEE identity to the completed unmasked V transport before
the subtraction. The identity exists specifically to keep a written binary64
result observable through XLA while retaining JIT and autodiff. No new
rounding helper or numerical method will be introduced.

## Frozen protocol

1. Extend the existing private round-153 arm with one default-off materialize-V
   hook. The candidate retains the raw-reference-depth prerequisite and the
   unmasked V transport; the control differs only by this materialization.
2. Reuse the round-146 production-JIT gate. Require the ordinary path, the
   explicitly false hook, and every prior prerequisite to remain bit-identical.
   Score substep-2 `transport_v`, `continuity_dv`, complete divergence, and
   `after_ssh` against the admitted rank-complete record.
3. If the complete causal chain is bit-exact, run the independent rung-0 and
   given-entry rung-7 kt=1..10 ladders at the frozen base and candidate. Compare
   all 200 rows per ladder, register every move and status change, and retain
   the first-non-bit checkpoint predicate.
4. The registered approximately 31 PSU compensation remains a hard veto: at
   kt=10 stage 3, neither ladder's salinity maximum may increase from its
   same-tree control.
5. A production landing is authorised only if the causal chain and both
   ladders pass. It additionally requires the shared GYRE ten-step and 30-day
   gates, DINO, LOCK_EXCHANGE, OVERFLOW, generic-card, citation, focused-test,
   and push-gate coverage. Otherwise production remains unchanged.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R154-P1 | The default-off materialization hook is passive. | Ordinary state and every exposed trace leaf are `np.array_equal` with the hook absent versus explicitly false. | Any default/control bit moves. |
| R154-P2 | Materializing the exact unmasked V transport closes the written downstream chain. | Substep-2 `transport_v`, `continuity_dv`, complete divergence, and `after_ssh` are each bit-exact; all earlier prerequisites remain exact. | Any target remains non-bit or an earlier exact prerequisite moves. |
| R154-P3 | The candidate sacrifices no exact ladder row and does not advance either first-debt checkpoint. | Both 200-row ladders complete, no exact row leaves exact, and each first non-bit checkpoint is unchanged or later. | Either ladder refuses, loses an exact row, or moves first debt earlier. |
| R154-P4 | The candidate is salinity-safe. | Each kt=10 stage-3 salinity maximum is no greater than its same-tree control. | Either maximum increases; the registered approximately 31 PSU exposure is an immediate veto. |
| R154-P5 | If production changes, every shared executing card passes its standing gate. | GYRE, DINO, tanks, generic cards, citations, and tests all pass their registered predicates. | Any unregistered or over-floor regression. |

## Controls and terminal rule

The existing operand-registry, one-ULP exact-cell, and round-153 causal plants
must still fire. Add a materialization-causal plant that disables only the new
boundary while preserving raw depth and unmasked transport; it must reproduce
the registered non-bit continuity chain and make the gate refuse. A candidate
that fails R154-P2 ends **HELD** without ladders. A candidate that passes the
chain but fails either ladder or salinity predicate also ends **HELD**, with
production unchanged.

ASKED choices: none. UNASKED choices: empty.
