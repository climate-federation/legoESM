# Preregistration — ORCA2 round 152 V metric-transport split

Date: 2026-10-05. Frozen base: `c00b179bc1`.
Claim class: rung-0 results are **independent**. No configuration, initial
state, forcing, sea-ice, stabiliser, carried-state form, threshold, or
`unmeasured_features` choice is authorised.

## Source statement and scope

NEMO forms the substep-2 V metric transport as the written two-product
statement
`zhV = e1v * va_e * zhvp2_e` over its extended V loop at compiled
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:568-570`, then uses
the north-minus-south difference in continuity and the sea-surface update at
`:584-591`.  The statement contains no face-mask multiplication.  The
resolved legoESM path calls `nemo_literal_metric_transports`, which evaluates
`((e1v * V) * H_v) * v_mask` at
`barotropic_latlon_cgrid.py:745-760`.

Round 151 proved that replacing the reconstructed reference V depth closes
the separate 30-cell midpoint-depth row but leaves the 68-cell metric
transport, V-difference, and sea-surface rows numerically unchanged.  On that
raw-depth arm, the recorded `va_e` and `zhvp2_e` are bit-exact; ownership among
`e1v`, the two written products, the extra compact face mask, and the compact
loop-domain association remains unassigned.

## Frozen protocol

1. Reuse the admitted round-95/97 substep record and the round-151 CPU
   production-JIT fp64/libm trace.  Reproduce observer passivity, the seven
   post-association arrays, all eight EEN coefficients, and the registered
   30/68/68/68 control census before scoring a new row.
2. Extend the existing round-146 gate and tests; do not build a second solver
   or a new callback.  The pre-implementation search found the needed trace
   operands and source-ordered transport helper there already.
3. On substep 2 of the raw-reference-depth arm, compare native `e1v`, `va_e`,
   and `zhvp2_e` separately against the admitted NEMO record over its complete
   V domain.  Preserve signed zeros in every comparison.
4. Replay, in order, `e1v * va_e`, then that result times `zhvp2_e`, using the
   shared source-rounding barrier after each written product.  Score this
   unmasked result and the production masked result independently against
   NEMO's recorded `zhV`.
5. Replay the recorded south-neighbour difference, divergence sum, and
   sea-surface statement with only the candidate V transport substituted.
   This is a statement replay, not a production causal arm or a landing.
6. Refuse a reordered operand registry and a one-ULP perturbation at an exact
   `zhV` cell.  No ORCA2 ladder or production landing is authorised unless the
   full causal chain later closes and the registered approximately 31 PSU
   salinity exposure is also closed.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R152-P1 | The registered prerequisites and round-151 control reproduce. | All prerequisite rows remain exact; raw-depth arm stays 0/68/68/68 for midpoint depth/transport/V difference/SSH. | Any prerequisite or registered count/max moves. |
| R152-P2 | `e1v`, `va_e`, and `zhvp2_e` are individually bit-exact on the full recorded V domain. | Each operand has 0 unequal cells. | Any operand is non-bit; the first such operand owns the walk instead. |
| R152-P3 | The source-written unmasked two-product is bit-exact and the extra compact `v_mask` creates the 68-cell transport debt. | Unmasked `zhV` has 0 unequal cells; the masked control reproduces 68 unequal cells at the registered maximum. | The unmasked replay remains non-bit or the masked replay does not reproduce production. |
| R152-P4 | Substituting only the unmasked `zhV` closes the recorded V-difference and sea-surface statements in replay. | Replayed `continuity_dv` and `after_ssh` each have 0 unequal cells. | Either row remains non-bit; ownership stays downstream or mixed. |
| R152-P5 | Production remains unchanged. | Only gate/test/docs files change; no `packages/` file changes. | Any model file or ordinary trajectory changes. |

## Controls and terminal rule

Both named plants must fire.  A matching offline replay names a candidate
statement but does not authorise a model edit.  The round ends **HELD** after
the split unless a separately preregistered production arm, both ORCA2
ladders, the salinity/tracer compensation gate, and the full shared-card gates
all pass within the same round.

ASKED choices: none. UNASKED choices: empty.
