# Preregistration — ORCA2 round 249 private-unit fold growth boundary

Date: 2026-10-11. Frozen base: `5ebe69712`. Scope: locate the origin of the
finite but explosive OMT-4 state exposed by the complete private fold/transport
unit. Results remain separately labelled **independent** and **given NEMO's
entry**. Measurement uses completed stage states and the already-established
barotropic trace only after bit-passivity checks. No in-executable observer,
deck, selector, carried state, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Cited program order

The executing oracle build constructs midpoint U/V and live face depths at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynspg_ts.f90:460-520`, transports
and SSH at `dynspg_ts.f90:530-558`, averaged external velocities at
`dynspg_ts.f90:563-580`, and the pressure/Coriolis/update path at
`dynspg_ts.f90:601-682`. NEMO then applies its post-stage boundary exchange to
U, V, T, and S at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stprk3_stg.f90:779-801`.

The private candidate is unchanged from rounds 245--248: the complete
round-217 fold/transport unit, raw NEMO surface V mask plus T-pivot north
neighbour, and the resolved D114 three-dimensional stage masks. It remains one
atomic diagnostic unit. Round 248 proved the kt=5 substep-2 zero-depth
reciprocal is downstream damage, so this round walks only finite kt=1--4
boundaries upstream of it.

## Frozen predictions and falsifiers

1. **R249-P1 — passive growth table.** For each label, production-OFF and the
   complete unit reproduce their ordinary completed states bit-for-bit under
   the stage-state observer. The unit completes kt=1--4 with finite SSH, U, V,
   T, and S. Any observer bit move or earlier non-finite value invalidates the
   table.
2. **R249-P2 — first growth boundary.** In compiled boundary order (kt entry,
   stage 1, stage 2, stage 3), report fold-band and interior max/RMS errors for
   SSH, U, V, T, and S, plus the exact fold-identity census. The first boundary
   whose maximum error grows by more than 10 times the immediately preceding
   boundary is the only boundary eligible for source-order replay. The frozen
   hypothesis is **kt=1 stage 1**, where the unit already worsens fold-band T/S
   by O(0.1--1) while improving SSH/U. A later boundary or no >10x growth
   **REFUTES** that hypothesis and becomes the result.
3. **R249-P3 — fold association signature.** If P2 names kt=1 stage 1, the
   first source-ordered substep or stage boundary with a non-rounding fold-band
   error also introduces a V-point fold-identity violation while NEMO's same
   field remains exact. This confirms the held Decision-116 in-loop or
   post-stage association as the owner candidate. If the first >10x boundary
   has no coincident fold-identity violation, or NEMO violates the same
   identity, the association hypothesis is **REFUTED** and the first recorded
   operand is named instead.
4. **R249-P4 — no landing.** This round is measurement-only. No package/model
   statement lands even if P3 confirms; the private candidate is removed and
   the result is HELD for the already-pending Decision 116. A downstream
   floor, clip, damping, or reciprocal change is forbidden.
5. **R249-P5 — controls.** Claim-label coverage, production-route identity,
   observer passivity, boundary order, finite kt=1--4 coverage, >10x owner,
   fold sign, NEMO fold identity, and citation plants must each refuse when
   violated.

## Required output

The receipt prints the kt=1 stage-1 OFF/unit/NEMO fold-band table first, then
the complete kt=1--4 boundary growth table, then the source-order replay at the
first >10x boundary. Every offline replay names the production D114 resolved
operand route. Decision 116 remains pending with the user and is not re-emitted.
