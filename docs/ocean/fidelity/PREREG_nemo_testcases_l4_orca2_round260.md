# ORCA2 round 260 preregistration — stage-2 pivot-V writer

Date: 2026-10-12. Frozen base: `f844b0ff2`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. The round replays kt=1
stage 2 from the exact completed stage-1 state of round 259. No deck, card
selector, carried-state contract, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Compiled order and existing instruments

NEMO advances the stage-1 velocity at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:468-482`, applies
the barotropic correction at `:510-539`, advances tracers at `:674-700`, and
then associates U, V, T and S together at `:779-802`. Stage 2 consumes that
associated Kmm velocity in the V transport product at `:269-285`, calls tracer
transport/advection at `:552-641`, and writes the thickness-weighted tracer
stage at `:674-700`. Under the compiled T-pivot V rule, the fold rows are
copied from the mirrored rows with sign -1 at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`.

The round reuses the committed round-256 post-stage association replay,
round-257 tracer-transport causal arm, round-258 five-operand V-product split,
and round-259 complete private unit. Any new diagnostic is offline from the
already-passive completed-stage trace. An in-executable observer is forbidden.

## Frozen predictions and falsifiers

1. **R260-P1 — provenance and reproduction.** Both labels run CPU
   fp64/libm production JIT from one clean committed tree through the
   production D114 resolved-mesh route. The private unit reproduces round 259:
   kt=1 stage-1 T/S at the campaign floor, followed by stage-2 fold-band S
   `6.817836451535307` PSU. A dirty tree, wrong route, non-passive trace, or
   failure to reproduce those rows invalidates all measurements.
2. **R260-P2 — first stage-2 input.** At the kt=1 stage-2 entry, legoESM's
   V velocity on the T-pivot row is zero or stale wherever NEMO's associated
   V velocity is nonzero. U is at the floor or differs only in signed zero.
   A valued earlier mismatch in the stage-1 completed T/S/SSH or in U refutes
   the predicted source order and redirects the offline split there.
3. **R260-P3 — cited writer.** Replaying NEMO's post-stage V association from
   the pre-association stage-1 velocity makes the pivot-row V entry bit-exact.
   The associated pivot row is copied from the mirrored row with sign -1; it
   is never a wall or an independently updated row. A nonzero residual after
   the literal copy refutes this statement as the owner.
4. **R260-P4 — causal closure.** Applying that association at every compiled
   post-stage call site inside the complete private unit makes the stage-2 V
   transport bit-exact and reduces the stage-2 fold-band T/S error to the
   `2e-10` campaign floor under both labels. If the entry V closes but T/S do
   not, split stage-2 `e1v`, live `e3v`, `vv(Kmm)`, `zvb`, and `vmask` in the
   written order before naming any later statement.
5. **R260-P5 — terminal predicate.** A closing unit has no greater-than-10x
   post-floor boundary through kt=1..7 and completes OMT-4 kt=8..10 under both
   labels without the live-W-thickness refusal. Failure names the first
   passive boundary and holds the unit.
6. **R260-P6 — Decision 96 and landing gates.** Only if P1-P5 pass, the whole
   indivisible unit may land when the first-over-bar row moves toward NEMO or
   is unchanged, a majority of RMS-moved rows move toward NEMO, no exact row
   is lost beyond the floor, and SSH maximum is not worse. Then run OMT-1..4,
   independent rung 0, given-entry rung 10, GYRE year, DINO month, tanks,
   citation gates, focused tests, the once-only fidelity battery, and separate
   read-only review. Every moved row is registered.
7. **R260-P7 — controls.** Exact worktree, both-label coverage, production
   route, passivity, round-259 reproduction, source order, pivot support,
   literal-copy closure, stage-2 tracer closure, terminal completion, and
   one-bit sensitivity each have a planted refusal.

## Stop conditions

No NEMO acquisition is planned. Missing passive operands are
`STOPPED_FOR_RECORD`; a surviving numerical boundary is `HELD`. No partial
association, mask, floor, clip, damping, or alternative configuration lands.

