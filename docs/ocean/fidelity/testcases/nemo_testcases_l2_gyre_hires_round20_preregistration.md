# GYRE high-resolution round 20 stage-1 boundary preregistration

Date: 2026-10-19. This document is committed before the common-state stage
records are scored or a new NEMO writer is run.

## Fixed state and providers

Scalar-math NEMO, vector-math NEMO, and legoESM start from the same
vector-math day-180 `r4_smag` restart, SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`,
and execute step 4321. The NEMO providers are the admitted round-17 scalar
and vector acquisitions; legoESM uses the certified CPU fp64/libm policy and
the production-JIT live-stage trace. Deck, timestep, forcing clock, masks,
restart bridge, and arithmetic flavours are unchanged from rounds 17--19.

Every result reports exact unequal-word count, maximum absolute residual, and
maximum ULP distance. The inherited scientific discriminator is the
round-16 four-times scalar/vector-NEMO envelope. A zero twin residual is
cleared by any nonzero legoESM residual. Exactness is still the fidelity bar:
clearing the four-times envelope localises a boundary but does not discharge
it. Trace passivity requires the ordinary and instrumented legoESM final state
to be byte-identical. Active-word plants must move the selected comparison.

For this source-boundary walk, the envelope statistic is the maximum absolute
residual over NEMO-owned wet U or V words, reported separately in source order
U then V. This is the maximum-residual diagnostic already used for the round-19
Kmm state (`2.71e-8` U and `2.41e-8 m/s` V); no regional or depth reduction is
introduced. The first stage or cumulative contributor is the earliest one
whose first face field clears four times the same field's scalar/vector
maximum. The all-word statistic is accompanied by exact counts and ULPs so a
large maximum cannot be described as exact or discharged.

## Ordered walk

The existing records first score corrected momentum after stages 1 and 2.
The earliest stage whose legoESM-minus-scalar residual clears four times the
scalar-minus-vector residual is the only stage eligible for a contributor
walk. If neither clears, the round stops without a writer request.

For stage 1, the executed source order is:

1. hydrostatic pressure gradient;
2. lateral diffusion;
3. total vorticity/Coriolis;
4. horizontal kinetic-energy-gradient advection;
5. vertical advection;
6. vertical average, bottom-drag initialisation, and surface stress into the
   two-dimensional forcing;
7. split-explicit external solve;
8. the stage-1 velocity update;
9. replacement of its depth mean by the external-mode target.

A new record is required only for missing boundaries in that order. For each
cumulative three-dimensional RHS boundary, the term is the difference from
the preceding boundary; the first term whose legoESM-minus-scalar residual
clears the four-times twin envelope becomes the factor-walk candidate. The
completed RHS, external-mode output, raw pre-correction stage state, and final
corrected stage state must close in source order before a term is named.

Vertical mixing has no stage-1 contribution: the compiled RK3 routine calls
the implicit vertical solve only in stage 3. This is an executed-call-graph
disposition, not a zero reconstructed by subtraction. RK3 performs pointer
swaps after stages 1 and 2 and no Asselin/filter arithmetic; the stage writer
therefore records the raw update and corrected output, not a fictitious
filter term.

## One-factor rule and scope guard

After a first differing contributor is measured, each factor arm must replace
one independently recorded NEMO input at that contributor's production
boundary. A transcription owner is named only if the substituted contributor
and downstream stage row both fall inside the scalar/vector envelope without
moving untouched rows. No shared-path edit is authorized by a smaller norm,
by a cumulative-row difference, or by the already measured stage-3 collapse
under the joint Kmm state.

This round may add only passive scorer/writer machinery, its tests and
receipt. It may request scalar and vector one-step acquisitions from the
pinned restart. It may not change physics, a deck, a threshold, or launch a
long run.
