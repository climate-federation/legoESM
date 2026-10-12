# ORCA2 round 251 preregistration — direct slow forcing and held fold associations

Date: 2026-10-12. Frozen base: `98e6c2b39`. Scope: OMT-4, both
**independent** and **given NEMO's entry** labels. This is an offline replay
and private-candidate measurement round. It changes no deck, selector,
carried-state contract, stabiliser, sea-ice field, or `unmeasured_features`
entry. The final package tree remains at the frozen base unless Decision 116
is separately unlocked; that decision is not re-emitted here.

## Compiled source order

The executing R230 build writes the completed depth mean, applies the
baroclinic drag, then the wind at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:198-230`.
The external solver copies that result, computes the two-dimensional
Coriolis trend, and subtracts it through the surface masks at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:286-324`.
The live external substep associates velocity, face depth, reciprocal depth,
and SSH together at `dynspg_ts.f90:721-730`; the RK3 stage associates u, v,
T, and S together at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-801`.

The round-250 result is the required starting fact: the literal drag
increment is exact, while the old 35-face `post_drag_v` field came from a
non-source-exact diagnostic override. Round 251 therefore forbids that
override and reads only the production chain's passive operands.

## Frozen predictions and falsifiers

1. **R251-P1 — direct-chain instrument.** The inherited slow-forcing and
   external-substep twins remain admitted, both labels agree, CPU/JIT/fp64 is
   stamped, and exposing the already-materialised operands moves no completed
   state bit. The private final-forcing override is absent. A record mismatch,
   label disagreement, dirty tree, wrong precision/backend, or any moved state
   invalidates the measurement.
2. **R251-P2 — direct slow forcing.** From the last source-exact boundary,
   the direct executing sequence `post_drag -> post_wind -> Coriolis ->
   surface-mask association -> final slow V` is bit-exact on the registered
   35 northern-fold faces under both labels. Any unequal row **REFUTES** this
   prediction and becomes the next source-ordered owner; the association
   candidate is then not scored.
3. **R251-P3 — association unit.** Only if P2 confirms, privately apply both
   held NEMO associations with the complete round-217 fold/transport unit:
   the seven-array external-substep association and the per-stage u/v/T/S
   association. The kt=1 stage-1 fold-band V maximum falls from
   `0.027227136128767482 m/s` toward the 2e-10 floor, real V fold-identity
   violations fall from about 1,300 to zero, the greater-than-10x kt=1
   growth disappears, and both labels complete kt=5 and kt=8. Failure of any
   predicate keeps the candidate private and names the first remaining
   boundary.
4. **R251-P4 — disposition.** Decision 116 still governs promotion of these
   held associations. Round 251 lands no package/model change. It records the
   direct-chain and private-unit measurements, retracts any falsified claim in
   the emitting gate, and ends HELD with the measured next boundary or the
   Decision-116-qualified unit.
5. **R251-P5 — controls.** Claim-label coverage, production operand route,
   absence of the final-forcing override, record admission, passivity, source
   order, 35-face support, direct-chain closure, boundary order, and one-bit
   fold plants must each refuse when violated. Receipt citations and their
   rigid-shift plant must fire.

## Stop conditions

No in-executable observer may be added. No NEMO run is requested: all operands
exist in admitted self-describing records and passive state outputs. No floor,
clip, damping, alternate configuration, carried-state change, or partial
association may land. If the direct chain is not exact, downstream association
scoring stops at its first unequal executable output.
