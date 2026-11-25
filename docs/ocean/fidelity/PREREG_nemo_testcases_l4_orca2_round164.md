# Preregistration — ORCA2 round 164 atomic V-transport landing

Date: 2026-10-07. Frozen base: `072fc5eb68`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round164/`.

Every rung-0 number is **independent**: the hierarchy card starts from its own
climatological T/S, zero velocity and zero sea surface. Every rung-7 number is
**given NEMO's entry** under Decision 52. The labels remain separate. Sea ice,
all six sea-ice selectors and the shipped card's `unmeasured_features` tuple
remain unchanged.

## Source unit and frozen protocol

The admitted rung-0 oracle reads raw `hu_0/hv_0` when constructing midpoint
face depths at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:519-545`, stores the
unmasked V metric transport at `dynspg_ts.f90:568-570`, consumes the stored
array in the separate continuity loop at `dynspg_ts.f90:584-591`, and
associates all seven external-mode fields at `dynspg_ts.f90:761-779`.
Rounds 146-154 proved those four prerequisites together at the substep-2
boundary. Round 155's first production attempt was retracted because the full
association exposed a compensating error and became non-finite at kt=8.
Rounds 156-163 then walked that exposure and named the missing atomic unit.

This round promotes the same four statements together on the existing
NEMO-literal barotropic path. No public selector is added. A literal card
missing raw face-depth operands refuses. Generic paths stay unchanged.

1. Restore the source-scoped production routing and require the production
   path to equal the existing complete private arm at the admitted boundary.
2. Run the independent rung-0 and given-entry rung-7 kt=1..10 ladders against
   round 163; register every moved row, exact-row loss, first-debt movement,
   and the kt=10 stage-3 S/SSH maxima.
3. Apply Decision 96 mechanically: the unit lands only if the first-over-bar
   row is toward or unchanged, a strict majority of moved RMS rows is toward,
   no certified exact row is lost, and SSH maximum is no worse.
4. If the ladder predicate passes, run the independent rung-0 month, GYRE
   ladder/year, DINO, VORTEX/tanks, card registry, citation gate and tests.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R164-P1 | Production equals the already-proved complete private arm at kt=1 substep 2. | V transport, V difference, divergence and SSH are bit-exact; explicitly enabling all retained arms changes no ordinary or exposed array. | Any named boundary is non-bit or an explicit arm moves a bit. |
| R164-P2 | The compensating unit now completes both ten-step ladders. | Both 200-row ladders complete with zero exact-row loss and unchanged-or-later first debt. | Refusal, any exact-row loss or earlier first debt. |
| R164-P3 | The atomic unit is a Decision-96 net improvement on independent rung 0. | Strictly more moved RMS rows go toward than away, first debt is toward/unchanged, no exact-row loss, and kt=10 stage-3 SSH maximum is no worse than `0.42832517646246693 m`. | Any predicate fails. |
| R164-P4 | The old halo salinity veto is removed by its measured consumer. | Rung-0 kt=10 stage-3 S maximum is no greater than the round-163 control `0.4156673855238111`; the dedicated veto passes. | S maximum increases or the approximately 31 PSU exposure returns. |
| R164-P5 | The exact V-transport chain advances the independent month beyond step 36. | First non-finite step is later than 36 or all 240 steps complete. | First non-finite stays at step 36 or moves earlier. |
| R164-P6 | Shared cards remain admitted. | GYRE satisfies Decisions 43/45/55/59/96, DINO stays below its fixed bar, tank/VORTEX registries preserve their standing statuses, and card/citation/test gates pass. | Any unregistered or over-bar regression. |

## Terminal rule

Any failure through R164-P4 leaves production unchanged and the round HELD
with the exact red row. R164-P5 may refute after a valid bit-exact statement
landing and then names the next independent-month boundary. No stabiliser,
configuration choice, sea-ice change, carried-state change, or bar relaxation
is authorised. Failed predictions remain in the receipt.

ASKED choices: the atomic compensation unit and Decision-96 predicate from the
round-163 OPEN item. UNASKED choices: empty.
