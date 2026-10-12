# ORCA2 round 262 preregistration — stage-1 V-ratio fold association

Date: 2026-10-12. Frozen base: `5e7ea0504`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. The round starts from
round 261's source-exact stage-2 V-transport split. No deck, card selector,
carried-state contract, stabiliser, sea-ice field, or `unmeasured_features`
entry may change.

## Compiled statement and atomic unit

NEMO computes the stage-after V-point free-surface ratio in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/domqco.f90:213-217`, associates
`r3va` with the V-point north-fold rule at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:168-170`, and only
then stores the stage-1 ratio at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:177-180`.
Round 261 measured that the missing association feeds both members of one
cancelling unit: stage-2 live `e3v(Kmm)` and the live `r1_hv` used by `zvb`.

The candidate adds only that cited association at the executing stage-1
ratio call site, using the existing tripolar fold primitive. It is measured
atomically with the already-committed private unit: the round-217 fold and
transport parts, raw V surface mask plus T-pivot neighbour, D114 resolved
three-dimensional stage mask, and the held in-loop/post-stage associations.
No member is eligible to land separately.

## Frozen predictions and falsifiers

1. **R262-P1 — source replay closure.** Under both labels, the associated
   stage-1 V ratio makes the round-261 rows exact: live `e3v` unequal count
   `668 -> 0`, live `r1_hv` `35 -> 0`, `zvb` `35 -> 0`, and stage-2 `zFv`
   `849 -> 0` on the same recorded supports. Any surviving unequal cell,
   wrong support, non-exact candidate/oracle replay, or moved source order
   refutes the candidate before a trajectory score.
2. **R262-P2 — stage-2 tracer consequence.** With the complete unit, kt=1
   stage-2 fold-band T/S are at or below the `2e-10` campaign floor under
   both labels. A value above the floor names a later source-ordered tracer
   boundary and keeps the unit private.
3. **R262-P3 — terminal closure.** The complete unit has no post-floor
   greater-than-10x boundary and completes every OMT-4 checkpoint through
   kt=10 under both labels. Any refusal, non-finite cell, missing checkpoint,
   or greater-than-10x boundary is **REFUTED** and the first boundary is kept.
4. **R262-P4 — Decision-96 predicate.** Only if P1-P3 pass, the atomic unit
   must move a majority of RMS-moved rows toward NEMO, move the first-over-bar
   row toward or leave it unchanged, lose no exact row beyond the floor, and
   not worsen the certified sea-surface maximum. Any failed predicate is
   **HELD**, with every touched row registered.
5. **R262-P5 — shared-card gates.** GYRE's certified ladder/year, DINO's
   certified month, and lock-exchange/overflow/tank gates are predicted
   byte-identical because their executing grids do not take ORCA2's active
   T-pivot fold route. The gates, not that reasoning, decide; any prohibited
   movement holds the landing.
6. **R262-P6 — controls.** Exact worktree, both-label coverage, compiled
   source order, record identity, candidate/oracle replay closure, the four
   round-261 rows, kt=1 stage-2 floor, kt=1..10 coverage, Decision-96 census,
   and one-bit sensitivity each have a planted refusal.

## Stop conditions

No NEMO acquisition is planned. A genuinely absent operand is
`STOPPED_FOR_RECORD`. A source replay that closes but a trajectory that still
refuses is `HELD` at its first passive boundary. No configuration choice,
floor, clip, stabiliser, sea-ice change, or partial cancelling member may land.

ASKED choices: the existing atomic unit and Decision-96 predicate.  
Unasked configuration choices made: none.
