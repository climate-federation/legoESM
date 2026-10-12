# ORCA2 round 263 preregistration — matched-point fold-V reconciliation

Date: 2026-10-12. Frozen base: `dbef7e692`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**. This round reconciles the
round-262 kt=1 stage-1 V number before any further source attribution. No deck,
card selector, carried-state contract, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Compiled statements and program points

NEMO builds stage-1 `r3va` in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:163-170`, passing the
V-point scalar to `lbc_lnk` with sign `+1._wp`, then stores it at
`:172-180`. The compiled T-pivot V rule copies the fold rows with the supplied
sign in
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/lbcnfd.f90:684-721`; this record ran
with `jpni=2`, so `lbclnk.f90:2105-2114` dispatches the same north-fold
association through `mpp_nfd`.

After each RK3 stage, NEMO associates the completed U/V/T/S stage fields at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-802`, with V sign
`-1._wp`. The admitted stage frame is written after that call. Round 262's
growth table instead scored legoESM's passive `stage_outputs`, which the live
trace currently materialises before any equivalent post-stage association.

## Frozen predictions and falsifiers

1. **R263-P1 — scalar V-ratio sign and rows.** Replaying the compiled `r3va`
   association with scalar sign `+1` on NEMO's stage-1 SSH makes the candidate
   live `e3v` and `r1_hv` valued rows exact under both labels, reproducing round
   262. A sign `-1` or a wrong source/target row must make the planted replay
   non-exact. Any valued residual with the source-exact `+1` rule refutes the
   round-262 association and requires a corrected private arm before proceeding.
2. **R263-P2 — matched program point.** The raw pre-association stage-1 V
   reproduces round 262's fold-band maximum `0.027227136128767482` m/s. Applying
   NEMO's post-stage V association before comparing with NEMO's post-association
   frame reduces that maximum to at or below the `2e-10` campaign floor under
   both labels. If it does, the round-262 “first valued boundary” is retracted
   as a program-point mismatch. If it remains above the floor, the stage-1 V
   update is a real model debt and owns the next walk.
3. **R263-P3 — next valued boundary.** If P2 confirms, the first valued
   boundary after matched association is expected at kt=1 stage 2, where the
   executable still consumes the unassociated stage-1 fields. The existing
   source-exact causal control — supplying the associated stage-1 U/V/T/S at
   the stage-2 entry — must reduce fold-band T/S to the campaign floor. Failure
   names the first later stage-2 statement instead; no association landing is
   inferred from prose.
4. **R263-P4 — instrument controls.** Both labels, clean-tree provenance,
   fp64/libm CPU JIT, passive live tracing, record identity, source order,
   scalar-sign sensitivity, row sensitivity, raw-boundary reproduction,
   matched-point closure, causal stage-2 closure, and one-bit sensitivity each
   have a planted refusal.

## Stop conditions

This is measurement first. If P1 fails, the corrected scalar association is
measured privately before any trajectory claim. If P2 confirms, Decision 116
still governs an association-alone landing; it is not re-emitted. A complete
atomic-unit landing remains eligible only after kt=1..10 completion and all
Decision-96 gates, which this round does not assume. A genuinely absent operand
is `STOPPED_FOR_RECORD`.

ASKED choices: the existing atomic unit and standing Decision-96 predicate.  
Unasked configuration choices made: none.
