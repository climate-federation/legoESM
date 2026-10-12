# ORCA2 round 256 preregistration — restart the private-unit fold growth walk

Date: 2026-10-12. Frozen base: `f29c9ff8e`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**, with every landed statement
through round 252 and the complete private fold/transport unit including both
held external-mode associations. This is an offline measurement round. No
deck, selector, carried-state contract, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Compiled statements and reused instrument

The round reuses and extends the committed round-249 passive stage-growth
instrument. It reads completed stage states only; a traced execution must be
array-identical to an untraced execution before any number is admissible.

The compiled OMT-4 oracle associates the seven external-mode fields after
every split-explicit substep at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:721-739`, then rotates
the U/V/SSH histories at `dynspg_ts.f90:781-793`. It associates the completed
three-dimensional U, V, T, and S fields after every RK3 stage at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`.
legoESM's private complete unit selects the literal external association
through `_NEMOWSRK3TestHooks.barotropic_external_mode_association`; it does not
turn any held statement into a public configuration choice.

## Frozen predictions and falsifiers

1. **R256-P1 — provenance and passivity.** Both labels run production JIT on
   CPU in fp64/libm from a clean exact commit, use the production D114 resolved
   mesh route, and the observed kt=1..7 completed states are bit-identical to
   their unobserved executions. A dirty or wrong tree, changed completed state,
   stale operand route, missing label, or non-finite NEMO frame invalidates all
   results.
2. **R256-P2 — first table before attribution.** The receipt prints the kt=1
   stage-1 fold-band maximum and RMS versus NEMO plus the exact fold-identity
   violation count for SSH, U, V, T, and S before naming any owner. It then
   prints every kt=1..7 stage boundary, split into fold band and interior, and
   names the first boundary whose maximum grows by more than 10x over the
   previous boundary with the `2e-10` floor.
3. **R256-P3 — association-placement discrimination.** If V already violates
   its fold identity at kt=1 stage 1 while NEMO's V residual is zero, the
   private external association is either placed before a later V writer or
   the per-stage U/V association is missing. Offline source-order replay must
   distinguish the stage V update from the compiled post-stage `lbc_lnk` before
   any code is eligible to land. If kt=1 stage 1 is fold-exact, this prediction
   is REFUTED and the first later stage/substep at which the identity breaks is
   the boundary to split instead.
4. **R256-P4 — source-ordered owner.** At the first identity-breaking boundary,
   score the raw completed V update, the external-mode in-loop/post-loop
   association where it executes, and the RK3 post-stage U/V/T/S association
   in compiled order. The first output that changes a zero NEMO fold residual
   to a nonzero candidate residual names the next statement. A replay that
   misses its own candidate or NEMO target invalidates the attribution.
5. **R256-P5 — disposition.** This measurement round is HELD. No partial unit,
   association, floor, clip, damping, or alternative configuration lands.
   Decision 116 remains pending and is not re-emitted.
6. **R256-P6 — controls.** Label coverage, exact worktree, production route,
   passivity, boundary order, NEMO fold identity, first-growth selection,
   source-order ownership, and one-bit detection each have a planted violation
   that must make the classifier refuse.

## Stop conditions

No NEMO acquisition is planned. A missing passive stage boundary or a replay
that cannot reproduce its target exactly makes the round
`STOPPED_FOR_RECORD`; it does not authorize an in-executable observer. The
shipped ORCA2 card and its SI3 `unmeasured_features` tuple remain unchanged.
