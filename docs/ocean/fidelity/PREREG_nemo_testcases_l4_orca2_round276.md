# ORCA2 round 276 preregistration — OMT early/late source-unit census

Date: 2026-10-14

Lane base: `2eaeaf515d818df52b1fd88413a299c5db7b8b6f`. Amendment (2026-10-29 provenance audit): the full SHA is `2eaeaf515d2fa69ccdcaba8e9fe6e17cd8a64959`; the tail of the value above was mistyped.

Primary label: **independent**.  Given-NEMO-entry rows are landing gates only
after a unit qualifies independently.  Counts always keep OMT rungs separate.

## Frozen question and cited statements

Round 275 found five source units that are individually net-toward on rung 0,
but their cumulative arm is majority-away on OMT-1, OMT-2, and OMT-3.  This
round measures the five units one at a time on OMT-1, then carries only
qualifiers through OMT-2 and OMT-3.  It separates early rows (kt=1..3) from
late rows (kt>=4) and records the first kt at which every away vote appears.

The five units are:

1. the atomic EEN ratio-fold and F-mask-pivot pair;
2. the stored T-point thickness-ratio product;
3. the literal stage T-ratio evaluation order;
4. the written stage-velocity update order;
5. NEMO's two vertical-momentum thickness divisors.

Two G2-certified multi-level partial-cell statements are added as singleton
arms, first on rung 0 and then on the same OMT protocol:

6. the tracer vertical solve's rounded live T thickness before its diagonal
   subtraction (`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/trazdf.f90:231-233`);
7. the barotropic forcing's ascending-level Fortran sums
   (`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/stp2d.f90:178-179`).

The vertical-momentum divisors are the executing statements at
`ORCA2_OMIP_L4_R230FOLDTRP/BLD/ppsrc/nemo/dynzdf.f90:191-205` and
`:360-375`.  The G2 implementations are imported from the certified geometry
ladder; they are not re-derived.  Every selector remains false by default and
sea ice plus `unmeasured_features` remain unchanged.

## Frozen predictions and falsifiers

### R276-P1 — complete singleton census

Prediction: all seven singleton arms construct on the current clean tree and
produce complete rung-0 and OMT-1 ladders.  Each comparison reports total,
early, and late RMS direction counts, exact-row losses, first-over-bar
direction, final SSH maximum, and the first kt of every away row.

Falsifier: an arm is omitted, inherits another selector, changes the protocol,
or produces an incomplete row set.  Such an arm is UNMEASURED-with-spec, not
silently classified.

### R276-P2 — module-dependent owner

Prediction: the written stage-velocity unit is the first of the original five
whose OMT-1 **early** rows are majority-away, because OMT-1 is the first vector
momentum rung and the one-level G1 proof exercised a different program.

Falsifier: its early rows are toward/equal, or another unit is the first with
an early majority away.  The actual first unit is retained and named.

### R276-P3 — late amplification discriminator

Prediction: at least one rung-0 qualifier that is majority-away over the full
OMT ladder has no away vote at kt=1..3; its first away vote appears at kt>=4.
That row is labelled late amplification, not a source-statement refutation.

Falsifier: every away-majority unit already has an early away majority.  Then
the interaction is module-dependent at the source boundary rather than a late
trajectory amplification.

### R276-P4 — G2 multi-level statements

Prediction: both newly imported G2 statements are net-toward on independent
rung 0 with zero exact-row loss.  On OMT rungs their early rows are
toward/equal because the same 30-level partial-cell statements execute there;
any away-majority is confined to late rows.

Falsifier: either statement loses an exact row, has an early away majority, or
does not execute through the production ORCA2 operand route.  It remains off.

### R276-P5 — landing and controls

A unit may land only if every claimed rung passes Decision 96: strict majority
toward among RMS-changing rows, no exact-row loss, and first-over-bar toward or
unchanged.  Late amplification is reported but does not itself override that
standing predicate without a user decision.  Any promoted shared statement
also requires rung 0, rung 10, OMT-1..4, GYRE year, DINO month, tanks,
citations, focused tests, the push battery, and the one allowed ocean-fidelity
battery.  A red predicate leaves all production selectors off.

The classifier's exact-loss and inherited-selector plants must fire.  The
citation shift plant must fire.  No configuration choice, carried-state
change, clip, floor, stabiliser, or sea-ice edit is authorised.
