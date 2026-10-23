# GYRE high-resolution round 19 formed-operand preregistration

Date: 2026-10-19. This document is committed before legoESM's live stage-3
ENE operands are exposed or any factor substitution is scored.

## Fixed state, providers, and exact bar

The comparison inherits round 18 without changing its deck, restart, step,
precision, mask, or reduction. Scalar-math NEMO, vector-math NEMO, and
legoESM start from the same vector-math day-180 restart (SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`),
execute `r4_smag` step 4321 under the certified fp64/libm CPU policy, and
score native surface V on the registered western-boundary wet-span row.
The primary bar remains the executed source accumulator: zero unequal words
between `NEMO_before + legoESM_term` and `NEMO_after`.

The admitted scalar NEMO operand record supplies q and `F_u`. Its admitted
stage-2 state supplies stage-3 Kmm velocity and SSH. The card's raw NEMO mesh
supplies the compiled metrics and Coriolis field. The existing shared QCO
builders form live U- and F-point thickness from the recorded SSH. A value is
never recovered by dividing the completed q or `F_u`; if a listed factor is
not available from these independent providers, the factor is UNMEASURED and
requires a new write-only NEMO record.

## Registered walk

First, the production stage-3 trace returns the exact q and metric transports
already consumed by the ENE stencil. The trace must be bitwise passive against
the ordinary step. Live q and live `F_u` are compared wordwise with scalar
NEMO over every source-consumed word. Zero unequal words means that formed
quantity is exact and its factor walk is skipped.

For a differing q, each arm changes one source factor and keeps scalar NEMO
`F_u` fixed so the completed-stencil comparison isolates q formation:

1. scalar-NEMO Kmm velocity in the relative-curl expression;
2. scalar-NEMO `ff_f` Coriolis;
3. scalar-NEMO live F-point thickness formed from Kmm SSH.

For a differing `F_u`, each arm changes one source factor and keeps scalar
NEMO q fixed so the completed-stencil comparison isolates `F_u` formation:

1. scalar-NEMO `e2u` metric;
2. scalar-NEMO live U-point thickness formed from Kmm SSH;
3. scalar-NEMO Kmm U velocity.

Each arm changes only the named factor inside legoESM's existing source-order
formation and then supplies that candidate formed operand through the already
validated round-18 q/transport seam. HPG and advection must remain bitwise
unchanged. A factor is **SUFFICIENT** only when both its candidate formed
operand and the registered source-accumulator row have zero unequal words.
A smaller residual is DEBT, not ownership. If no single factor is sufficient,
the result is a coupled composition boundary and no physics edit is allowed.

The non-vacuity controls plant one consumed word in the live q and `F_u`
outputs and require the corresponding registered V row to move. The ordinary
trace and the unplanted factor walk must use a clean committed tree.

## Executed source order

The acquired writer's compiled `stprk3_stg.f90:404-430` calls `dyn_hpg`, then
`dyn_vor(kstp,Kmm,...)`, then vector-form `dyn_adv`; its record header fixes
stage 3 to Kmm=2. The selected `np_CRV` arm forms Coriolis plus the relative
curl at compiled `dynvor.f90:552-560`, divides by live F thickness at
`:571-573`, forms `F_u=(e2u*e3u(Kmm))*u(Kmm)` at `:574-578`, and consumes q
and `F_u` in the V accumulator at `:584-591`.

## Scope guard

This preregistration authorizes the smallest extension of the existing
write-only live-operand trace and round-18 scorer. A shared-path arithmetic
change is eligible only after a single factor clears the exact formed-operand
and source-accumulator bars and the executed NEMO source proves a transcription
difference. No deck, public option, threshold, or long run is authorized.
