# ORCA2 round 257 preregistration — reconcile and split the first tracer writer

Date: 2026-10-12. Frozen base: `1f5678ebb`. Scope: OMT-4, separately
labelled **independent** and **given NEMO's entry**, with every landed statement
through round 252 and the complete private fold/transport unit used by round
256. This is an offline measurement round. No deck, selector, carried-state
contract, stabiliser, sea-ice field, or `unmeasured_features` entry may change.

## Reconciliation before attribution

Round 249 measured commit `5ebe69712`; round 256 measured the later tree at
`f29c9ff8e`. The label pair, three-row fold band, physical T-pivot half, and
private hook set are otherwise the same. Between those trees the executing
private unit gained the resolved slow-forcing masks and literal drag addition
(`920f52b3f`, `4a2195086`, with the scoping repair `ca909a37a`). Therefore the
round-249 exact stage-1 T/S row and round-256 `0.17735225873138316 K` /
`3.283688696376231 PSU` row are not measurements of one executable. Round 257
must print this comparison mechanically before using either row. The current-
tree measurement stands if it reproduces round 256 under both labels; otherwise
both claims remain unresolved and no source attribution is allowed.

## Compiled source order and reused records

The executing record build is
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo`. At stage 1 it builds `zub/zvb` and
metric transports at `stprk3_stg.f90:269-285`. The FCT card dispatches stages
1 and 2 to CEN2; the horizontal face products and divergence are
`traadv_cen.f90:149-161`, the vertical CEN2 flux/divergence is
`traadv_cen.f90:202-227`, and the stage update is
`stprk3_stg.f90:662-685`. The completed stage is associated at
`stprk3_stg.f90:779-802`.

The admitted round-230 record already contains the rank-0 stage-1 `zFu/zFv/zFw`,
the after-advection and after-SBC accumulators, the Kbb/Kmm/Kaa tracers, and all
three r3t levels. The measurement reuses the production literal helpers and
the passive completed-stage trace; it adds no in-executable observer and asks
for no NEMO acquisition.

## Frozen predictions and falsifiers

1. **R257-P1 — provenance, labels, and reconciliation.** Both labels run CPU
   fp64/libm production JIT from this clean exact commit, use the D114 resolved
   mesh route, and reproduce round 256's stage-1 physical-half values. The
   historical/current label, support, hook, and tree stamps are printed. A
   label/support/hook mismatch, a dirty tree, or failure to reproduce round
   256 invalidates the round rather than choosing one receipt post hoc.
2. **R257-P2 — source order.** On the physical T-pivot half, score in this
   order: Kmm T/S; `zFu`, `zFv`, `zFw`; U and V CEN2 face products; horizontal
   and vertical divergence; after-advection accumulator; after-SBC
   accumulator; qco stage update; post-stage association. The first non-bit
   row owns the walk. Every earlier row must be exact or explicitly declared
   unmeasured.
3. **R257-P3 — transport exposure hypothesis.** The first numerical debt is
   predicted to be the stage-1 V transport `zFv` at the T-pivot north face,
   not the CEN2 multiplication itself. Substituting NEMO's recorded transport
   into the already-certified literal CEN2 replay must reduce the physical-
   half T/S maxima to the round-249 scale (T <= `0.0013611108 K`, S <=
   `0.0009604808 PSU`). If all three transports are exact, or recorded
   transports do not reduce the boundary, this prediction is **REFUTED** and
   the first later arithmetic row is named instead.
4. **R257-P4 — pivot-face discrimination.** If `zFv` is first, report its
   unequal count, maximum, argmax, and sign/ratio on the physical-half pivot
   faces. Distinguish a wrong transport operand from a double/misoriented T
   update by replaying NEMO's `0.5*pV*(T_j+T_j+1)` statement. A replay that
   misses either the candidate's own accumulator or NEMO's recorded
   accumulator invalidates the attribution.
5. **R257-P5 — disposition.** Measurement only. No partial unit, association,
   fold rule, floor, clip, damping, or configuration change lands. Decision
   116 stays pending and is not re-emitted.
6. **R257-P6 — controls.** Exact worktree, both-label coverage, support, source
   order, current-tree reproduction, first-owner selection, recorded-operand
   closure, and one-bit sensitivity each have a planted violation that must
   make the classifier refuse.

## Stop conditions

If the rank-0 record does not cover the physical-half argmax, if a required
operand is absent, or if offline replay cannot reproduce both its candidate
and NEMO endpoints, the result is `STOPPED_FOR_RECORD`. It never authorizes an
in-executable observer. The shipped ORCA2 card and its SI3
`unmeasured_features` tuple remain unchanged.
