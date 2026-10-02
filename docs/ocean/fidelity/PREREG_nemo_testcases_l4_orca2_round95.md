# Preregistration — ORCA2 round 95 rung-0 split-explicit substep record

Date: 2026-10-02. Base: `0bfe1f676`. Every model number produced by this
round is labelled **independent**: the rung-0 state comes from NEMO's admitted
from-rest stage-0 frame.

## Frozen measurement

Before this file was committed, the compiled rung-0 `dyn_spg_ts` source, the
round-93 target's filenames and sizes, and the VORTEX round-196/197 instrument
and receipts were read. No legacy ORCA2 barotropic payload was decoded or
compared with legoESM.

The round will produce:

1. A rank-complete, self-describing kt=1 record of the rung-0 external-mode
   loop. Its header owns `icycle`; every array owns its name, rank and extents;
   the checker derives payload lengths and required frames from those headers.
2. Admission only if both ranks cover the 148x180 global domain once, every
   substep frame is present, values are finite, exact EOF is reached, and all
   twenty kt=1..10 terminal restart shards remain byte-identical to the
   admitted round-93 parent.
3. An independent CPU/fp64 walk in compiled order: loop-entry forcing and
   carried fields; mid-step extrapolation; face depth and transport; after-SSH;
   pressure-gradient; `dyn_cor_2D`; drag; velocity update; and filtered exit.
   The first active non-bit boundary owns the next round.
4. Header, field-name, missing-frame, truncation, swapped-rank,
   restart-byte, and observer-passivity controls. Every applicable plant must
   refuse before any model number is admitted.

The compiled rung-0 program copies the forcing at
`ORCA2_OMIP_L4_R93SLOW/BLD/ppsrc/nemo/dynspg_ts.f90:287-291`, initializes the
external state at `:339-381`, runs the substep loop at `:446`, forms the
mid-step velocity at `:460-485`, the face depths at `:487-520`, transport and
after-SSH at `:530-558`, pressure gradient at `:601-616`, Coriolis trend at
`:618-623`, drag at `:633-653`, and updated velocity at `:655-710`.

## Predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R95-P1 | The inherited ORCA2 `oracle_bt_*` streams are not admissible for this walk. | They are root-only, have no rank tag, and use legacy fixed-layout headers rather than per-array self-description. | Two complete rank-tagged, self-describing streams already exist: **REFUTED**; admit those without rebuilding. |
| R95-P2 | A VORTEX-pattern additions-only writer can record rung 0 without moving the trajectory. | All terminal restart shards are byte-identical to round 93. | Any shard moves: **REFUTED**; withhold the record and repair the observer. |
| R95-P3 | The first active non-bit boundary is no later than the first substep's after-SSH. | At least one boundary from loop entry through after-SSH is non-bit at substep 1. | All are bit-exact: **REFUTED**; continue in source order through pressure gradient, Coriolis, drag and velocity update. |
| R95-P4 | ORCA2's barotropic `e3f_0vor` denominator differs from the card's frozen coefficient at a land-adjacent vertex, as on VORTEX. | The cross-operator check with NEMO's mid-step velocity is non-bit only on the land-adjacent ring and the NEMO four-cell masked denominator makes it bit-exact. | Any broader/different support, or failure to close exactly: **REFUTED**; do not attribute or land. |

## Landing and refusal bar

This is an acquisition/measurement round. No model statement lands unless a
one-variable cited arm is bit-exact at its boundary and passes the ORCA2 rung-0
ladder plus every shared-card gate. The held round-94 reduction arm is not
mixed into this record. No selector, configuration choice, stabilizer,
carried-state convention, sea-ice field, threshold, or
`unmeasured_features` entry changes.
