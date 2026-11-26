# Preregistration — ORCA2 round 246 atomic V-mask / north-neighbour unit

Date: 2026-10-11. Frozen base: `c1b476175`. Scope: OMT-4 first, then the
standing Decision-96 landing gates only if the direct and kt=8 predicates
pass. Results remain separately labelled **independent** and **given NEMO's
entry**. No deck, selector, carried state, stabiliser, sea-ice field, or
`unmeasured_features` entry may change.

## Cited source unit

The executing OMT-4 build forms `ssvmask` from the vertically reduced raw
V-point mask after NEMO's V-point fold association at
`ORCA2_OMIP_L4_R210OMT1_P3/BLD/ppsrc/nemo/dommsk.f90:206-249`. It uses that
mask in the vector V update at `dynspg_ts.f90:665-679`. At the next substep it
forms the V midpoint depth from the raw reference depth, the two T-point
surface terms, and `ssvmask` at `dynspg_ts.f90:487-520`, then materialises
`zhV=e1v*va_e*zhvp2_e` at `dynspg_ts.f90:530-535`. The north T-point operand
on the T-pivot halo is the `T/W` association in `lbcnfd.f90:945-959`.

NEMO exposes no selector among those statements. The measurement candidate is
therefore one atomic unit: raw NEMO V surface-mask routing plus the T-pivot
north-neighbour association, together with the complete round-217 literal
external-mode unit (raw reference face depths, literal fold association,
unmasked/materialised V transport, and the vector slow/update mask pair).
Partial members are diagnostic arms only and cannot land.

## Frozen predictions and falsifiers

1. **R246-P1 — direct closure.** The production-shaped candidate reproduces
   round 245's direct table for both labels: midpoint V, midpoint V depth, and
   `zhV` are each 0 unequal at substeps 3, 4, 5, and 6. Any nonzero row
   **INVALIDATES** the candidate before a trajectory is scored.
2. **R246-P2 — OMT-4 survival.** With the complete atomic unit, OMT-4 completes
   kt=8 and continues through its admitted kt=10 ladder. The former
   `raw-mesh e3w_int must contain only finite values > 0` refusal disappears.
   Any refusal at kt=8 or earlier **REFUTES** P2; retain its first exact
   boundary and do not run landing gates.
3. **R246-P3 — external V operands.** On the five-operand `zFv` table, `e1v`,
   live `e3v`, `vv`, and `vmask` remain exact, the carried external correction
   collapses with the now-exact fold transport, and `vn_adv` falls from the
   registered 919 unequal columns to at most 35. Fold-band salinity falls from
   3.283 PSU to at most 0.0012 PSU. A non-exact geometry/mask row or unchanged
   `vn_adv` **REFUTES** the corresponding ownership claim.
4. **R246-P4 — Decision 96.** Only a candidate satisfying P1-P3 proceeds to
   OMT-1..4, independent rung 0, given-entry rung 10, GYRE, DINO, tanks, and
   moved-row gates. It lands only if the first-over-bar row is toward or
   unchanged, a majority of RMS-moved rows are toward NEMO, no exact row is
   lost beyond the floor, and the SSH maximum is not worse. Otherwise it is
   **HELD** at the first red predicate and production is restored.
5. **R246-P5 — controls.** Label coverage, direct closure, kt=8 survival,
   moved-row census, and source/citation plants must each refuse. The model
   change must preserve JIT, autodiff-visible array operations, and GYRE's
   analytic-mesh fallback.

## Stop conditions

No configuration choice or NEMO acquisition is authorised. Decision 116 is
pending separately and is not part of this atomic unit. A failed prediction is
kept as a refutation, never repaired by a stabiliser or by relaxing a gate.
