# Preregistration — NEMO testcase L2 GYRE round 145

Date: 2026-09-21

Incoming lane tip: `296bba7d39732989248d64fed250c5b134e9f604`

This document is frozen before changing production arithmetic or measuring a
new arm. Evidence will live under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round145/`.

Round 144 proved through the production-JIT step from NEMO's admitted day-180
entry that the wind statement's live inverse depth is its first non-bit input.
Replacing only that U/V pair by the recorded NEMO values makes both incoming
slow-forcing rows BIT: U changes from 580 / 580 unequal at
`1.3293041661996061e-13 m s-2` to zero and V changes from 570 / 570 unequal at
`1.0882954083811340e-13 m s-2` to zero. Density and stress substitutions are
inert. This round tests the one-variable production routing named in the
round-144 OPEN section.

## P0 — compiled statement and one-variable candidate

The admitted compiled program constructs the stored reference reciprocals at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domain.f90:212-215`, constructs the
step-entry QCO U/V ratios at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:160-182` with the
surface-area-weighted statements at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/domqco.f90:188-218`, and consumes
`r1_hu_0/(1+r3u(Kbb))` and `r1_hv_0/(1+r3v(Kbb))` in the wind additions at
`GYRE_OMIP_L2_P3_SM_R140RHS/BLD/ppsrc/nemo/stp2d.f90:241-260`.

The candidate must reuse the existing shared QCO face-geometry evaluation at
the step-entry SSH and route its already available reciprocal outputs to the
wind statement. It may not add a formula, selector, state field, configuration
choice, or stabilizer. The momentum-advection face thickness continues to use
the same helper result. The only production-dataflow difference is the source
of the two wind inverse-depth operands.

Before measuring the candidate, an independently stamped clean base clone at
this preregistration commit must reproduce the ladder, month, year, DINO, and
generic-card controls. A source-tree or protocol difference besides this
routing REFUTES the controlled comparison.

## P1 — local production proof and plants

The round-144 production-JIT walk must first reproduce its ordinary post-drag
plus live-wind rows exactly. With the production candidate, both incoming U/V
rows are predicted BIT: 0 / 580 U and 0 / 570 V cells unequal. The downstream
final U/V rows may retain only the already registered Coriolis residue.

The callback/plain identity remains BIT across all 23 returned-state leaves
and the actual external-call operands. A production plant changes one consumed
wet QCO reciprocal by one ULP and must move a registered incoming row through
the complete step, print `STATUS PLANT-FIRED`, and exit nonzero. A closed-row
registry omission plant must also exit nonzero. Any non-bit incoming row,
default-path discrepancy, missing propagation, or unregistered local row
REFUTES local exactness and stops trajectory measurement.

## P2 — frozen trajectory predictions and landing falsifiers

The clean same-tip base is predicted to reproduce these current production
headlines: kt2 T/S maximum absolute differences
`1.4210854715202004e-14` / `2.1316282072803006e-14`, kt2 U/V RMS
`2.7377110452773967e-12` / `3.2849219221489645e-12`, kt3 T/S RMS
`8.659373840202989e-7` / `7.027291104577671e-8`, day-30 T3D RMS
`6.890431487825909e-5 K`, day-240 T3D RMS
`1.6446740233175394e-2 K`, and day-360 T3D RMS
`1.1223571247843664e-2 K`. Failure to reproduce is an instrument disagreement
that must be reconciled before interpreting the candidate.

Frozen candidate prediction: kt1 rows remain AT-BAR; first-over-bar remains
kt2 U/V; kt2 T/S remain AT-BAR; kt2 U/V decrease or remain unchanged; kt3 T/S
and every later moved row are registered. Day-30, day-240, and day-360 T3D RMS
each decrease. The exact predicted upper bounds are therefore the base values
above, with day 30 strictly below its base and days 240/360 at most their base.

The candidate lands only if all Decision-43/45 conditions pass: day 30 strictly
decreases; first-over-bar is not earlier; no kt1 AT-BAR row leaves; every moved
70-row ladder field is exactly registered with before/after values; day 240 and
day 360 do not worsen; all eight year rows are registered; and every executing
certified card is measured. A failed numerical prediction remains REFUTED in
the receipt. If any landing condition fails, restore production and preserve a
held patch rather than weakening a bar.

## P3 — DINO and other certified cards

The source statement is shared, but the resolved shipped DINO cards currently
set `surface_stress_implicit=False` and therefore do not execute this legoESM
wind branch. That execution claim must be instantiated, not inferred. Run the
same DINO card suite before and after and compare a production-step endpoint;
the frozen prediction is bit-identical. A changed DINO bit is a registered
movement and blocks landing pending attribution.

GYRE-zco and the generic NEMO-GYRE recipe are predicted to execute the route;
the generic recipe's certified three-step endpoint must be measured before and
after and every moved field registered. LOCK_EXCHANGE-zco and OVERFLOW-zps are
predicted not to execute because both resolve `surface_stress_implicit=False`;
their route predicates must be checked from the recipes. ORCA2 remains
`UNMEASURED-WITH-SPEC`: its separate ocean-only lane must instantiate the
resolved surface-stress and QCO program, then compare its own ladder before
this GYRE result is generalized.

## P4 — review, citations, and stop condition

A separate read-only Codex pass must try to refute helper ancestry, time level,
native-to-redundant face mapping, one-variable isolation, production-JIT local
closure, DINO execution classification, complete card coverage, moved-row
registry, and month/year admission. `DO NOT SHIP` blocks the diff. Every
compiled-source citation is mapped by the receipt citation gate; its shifted
citation plant must exit nonzero.

No configuration, default, carried state, scheme, canonical NEMO source, or
immutable bar changes. `ACQUISITION_NEEDED` and `DECISION_NEEDED` are expected
to be `NONE`.
