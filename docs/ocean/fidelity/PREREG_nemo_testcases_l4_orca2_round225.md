# Preregistration: ORCA2 round 225 — OMT-4 nonosc fold walk

Date: 2026-10-10. Frozen base: `730ae56be`. Scope: continue round 224's
source-ordered OMT-4 stage-3 FCT walk through the live `nonosc` north-neighbour
stencil, guarded beta budgets, and V-face coefficient. This file is committed
before running a round-225 replay or trajectory.

No NEMO run, configuration choice, carried-state change, stabiliser, sea-ice
selector, threshold, or `unmeasured_features` change is authorised. The
round-224 donor/centred face arm stays retracted unless it forms part of the
complete atomic unit with a newly named limiter statement. Replays are CPU,
JIT, fp64/libm and consume only the admitted completed stage state; no
in-executable observer is permitted.

## Compiled source order

The resolved live limiter is `nonosc`, not `nonosc_org`, called at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/traadv_fct.f90:306-316` and defined
at `traadv_fct.f90:743-938`. It constructs masked per-cell upper/lower bounds
at `:798-847`, reads the seven-member north-neighbour stencil at `:849-861`,
forms `zpos/zneg/zbt` and guards both divisions at `:862-878`, then selects the
V-face coefficient at `:888-915`. The provisional tracer is T-point exchanged
before the call at `traadv_fct.f90:306-316`; the compiled T-pivot exchange is
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/lbcnfd.f90:581-638`.

The existing production trace registries already expose the seven stencil
members and the literal beta operands. Round 225 extends that one instrument;
it does not create a second limiter transcription.

## Frozen predictions and falsifiers

1. **R225-P1 — first limiter difference is the north stencil.** CONFIRM: on
   both T and S, all six non-north members of the top-row seven-member stencil
   remain bit-identical while replacing the copied northern row with NEMO's
   T-pivot source changes at least one active fold cell's `zup` or `zdo`; no
   changed cell lies off the fold row. REFUTE: an earlier member differs, the
   literal north source is inert, or any change lies off-fold.
2. **R225-P2 — guards are live and source-faithful.** CONFIRM: the recorded
   fold cells are classified by the exact source predicates
   `zup != -HUGE && zpos != 0` and `zdo != HUGE && zneg != 0`; every guarded-off
   division yields the no-clip sentinel and remains finite, while a planted
   unguarded division changes a finite/non-finite mask. REFUTE: the live code
   divides at a guarded-off cell, a source guard has no active example, or the
   plant is inert.
3. **R225-P3 — V coefficient is downstream of the folded budgets.** CONFIRM:
   the literal north-fold V-face coefficient uses the source-selected adjacent
   beta pair and differs from the neutral wall coefficient on at least one
   active nonzero antidiffusive fold face; the sign-selected pair reproduces
   the literal coefficient bit-for-bit. REFUTE: the coefficient is unchanged,
   the selected pair does not reproduce it, or a non-fold face moves.
4. **R225-P4 — limiter statement sufficiency.** CONFIRM: a private atomic arm
   containing round 224's complete face association plus the first differing
   limiter statement removes or delays the OMT-4 kt=8 live-W-thickness refusal;
   then score the whole unit under Decision 96. REFUTE: the refusal remains at
   the identical boundary/log signature. On refutation, retract the arm and
   name the next source-ordered limiter statement; do not manufacture a census.
5. **R225-P5 — controls.** Plants for north-source association, fold-only
   support, guard activity, coefficient selection, and statement sufficiency
   must each refuse. A plant that stays green invalidates its claim.

## Stop conditions

If the admitted state cannot support the existing passive trace, status is
`STOPPED_FOR_RECORD` with the exact missing operand. If the first statement is
named but no atomic candidate completes the shared gate, status is `HELD` and
the failed prediction remains in the receipt. OMT-5 stays blocked until the
OMT-4 refusal is explained.
