# Preregistration — ORCA2 round 93 rung-0 stage-1 RHS walk

Date: 2026-10-02. Base: `da33d419a`. Every model number produced by this
round is labelled **independent**: the rung-0 state comes from NEMO's admitted
from-rest stage-0 frame, not from the shipped ORCA2+SI3 entry.

## Frozen measurement

Before this file was committed, the operator-reported round-92 acquisition was
re-admitted without modification. Its two self-describing rank records cover
the 148x180 domain exactly once, contain the ten HPG/LDF/VOR/KEG/ZAD cumulative
momentum arrays at kt=1, and preserve all 20 terminal restart shards byte for
byte. No record payload was compared with legoESM before this preregistration.

The repository was searched before implementation. The existing GYRE round-84
walk already defines the compiled source-order accumulation and first-non-bit
selection, the round-92 admission gate already parses the self-describing
record, and the round-92 rung-0 gate already builds and bridges the independent
measurement card. Round 93 will reuse those implementations.

The round will produce:

1. One full-domain comparison of legoESM's production-JIT stage-1 cumulative
   U/V accumulators against NEMO after HPG, LDF, VOR, KEG, and ZAD, in the
   compiled order in `stp2d.f90`.
2. A mechanical first-non-bit statement, including unequal-cell count, maximum
   absolute residual, RMS residual, and argmax location. The NEMO `jpk` bottom
   slot is separately required to be exact zero rather than silently dropped.
3. A trace-passivity comparison between the ordinary production step and the
   write-only live-operand trace, plus one-bit, layout, and boundary-order
   controls that must refuse.
4. If HPG is first non-bit, an operand census using only already-admitted
   records. If those records do not bracket its internal statements on both
   ranks, the owner stays UNMEASURED and a new acquisition is written rather
   than inferred.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R93-P1 | The first non-bit cumulative statement is HPG. | At least one of `after_hpg_u`/`after_hpg_v` is non-bit; entry remains bit-exact and source order is mechanically verified. | Both HPG faces are bit-exact: **REFUTED**; continue in compiled LDF/VOR/KEG/ZAD order. |
| R93-P2 | Later accumulators do not precede the first HPG mismatch. | The gate names the first boundary by compiled boundary then U/V face order, independent of residual magnitude. | A later or larger row is named first while HPG is non-bit: instrument failure; refuse. |
| R93-P3 | The live trace is write-only. | Ordinary and traced final T/S/u/v/ssh arrays are bit-identical. | Any final-state bit moves: **REFUTED**; no statement claim. |
| R93-P4 | The admitted record is sufficient to name a cumulative statement but not necessarily an internal HPG line. | The first accumulator boundary is named; an internal owner is named only if rank-complete admitted operands bracket it. | Missing rank coverage, a nonzero `jpk` slot, or absent internal operands: stop at the exact missing stream and write an acquisition. |

## Landing and refusal bar

This is a measurement round. No model or card selector lands merely because a
cumulative boundary differs. A model statement is eligible only after its own
inputs and output are compared bit-for-bit and its path is proven active in the
compiled rung-0 configuration. The shipped ORCA2 card, its sea-ice debt tuple,
all configuration choices, stabilizers, thresholds, and carried-state
conventions remain unchanged.
