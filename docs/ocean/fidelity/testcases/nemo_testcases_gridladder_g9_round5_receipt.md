# Grid ladder rung G9 — round 5 receipt (gridladder_r50)

Status: **LANDED; G9 NOT CERTIFIED.**  G9 now selects a literal arithmetic
form of the shared FCT pipeline.  The kt-1 stage-3 FCT rate moves from
404,775 T / 415,895 S unequal wet cells to 28 / 17 and is at the bar, but the
exact-closure preregistration is refuted.  The independent and two-step score
still first exceeds the `1e-15` bar at kt 2.

Record: round 46's admitted `oracle_g9_r46`, unchanged.  No acquisition.
Preregistration commit: `3b43cf520`.

- codex review: operator
- independent review: operator (Claude)
- evidence: `gridladder_rounds/round50/`
- implementation: G9-selected; every other card keeps the generic FCT graph

## 1. Compiled statement and implementation

The active two-step upstream program writes its first faces, midpoint,
averaged faces and full upstream result in compiled order
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:495-609`).  The selected
`nn_fct_h = nn_fct_v = 2` arms then form the centred high-order horizontal
and vertical faces
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:193-197,260-266`), `nonosc`
limits their antidiffusive difference
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:316,743-938`), and the final
write adds the limited correction
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:320-329`).

The shared `fct_tracer_advection` has one static, selectable arithmetic form:
the ordinary form still consumes metric-free mass fluxes, while the literal
form consumes the already metric-bearing stage `(pU,pV,pW)` triplet and keeps
NEMO's array-store association through the same upstream, high-order,
limiter, fold and correction pipeline.  G9's existing
`rk3_stage3_tracer_krhs_nemo_order` route selects it at stage 3.  No second
FCT implementation and no new card field were added.  Rounds 47-48's fold
operations remain in the shared path.

Blast radius: masked-address step jaxprs are unchanged from round 49 for G5
`339a8016d262c4f6` (length 7,867,009) and G8 `03ccdb94113eab61` (length
8,074,511).  G9 alone reaches the new static branch.

## 2. Preregistered kt-1 verdicts

All comparisons are CPU, fp64, jitted, with traced state/forcing or traced
literal operands.

| preregistered quantity | before | prediction | observed | verdict |
|---|---:|---:|---:|---|
| after-`tra_adv` Krhs T / S unequal wet cells | 404,775 / 415,895 | 0 / 0; 1 refutes | 28 / 17 | REFUTED |
| pre-implicit content T / S | 1,226 / 762 | 0 / 0 | 613 / 248 | REFUTED |
| step-end all-level T / S | 1,076 / 587 | 0 / 0 | 533 / 202 | REFUTED |
| G5 / G8 jaxpr hash | `339a8016d262c4f6` / `03ccdb94113eab61` | unchanged | unchanged | CONFIRMED |

Before `tra_adv`, the card's stage-3 Kmm and Kbb T/S inputs are bit-identical
to the round-46 record: 0 / 0 unequal wet cells for each tracer.  The record's
pre-advection Krhs is zero and the card initializes the same zero accumulator.
After `tra_adv`, the remaining 28 T / 17 S unequal rates have normalized
maxima `9.44e-18` / `1.26e-17`, both at the bar.  Thus the source statement is
faithfully selected but the preregistered exact result is not claimed.

The post-advection content comparison has 107,773 T / 107,458 S unequal wet
cells with normalized maxima `1.38e-16` / `2.09e-16`; the later source-ordered
pre-implicit content has 613 / 248 at `6.92e-17` / `5.21e-17`.  Supplying
NEMO's recorded pre-implicit content closes the implicit solve and step end
exactly (0 / 0), so the remaining step-end 533 / 202 cells are upstream of
that solve.  The record content regression now asserts only the value bar,
not any fitted unequal-cell count.

## 3. Mandatory G9 score, both labels, kt 1..10

Rows are ssh, uu_b, vv_b, u, v, T and S.  Counts are exact / non-exact
AT-BAR / DEBT; the last column is the worst normalized row.

| label | kt | exact / AT-BAR / DEBT | worst |
|---|---:|---:|---:|
| independent | 1 | 5 / 2 / 0 | T `2.361e-16` |
| independent | 2 | 0 / 3 / 4 | v `2.385e-14` |
| independent | 3 | 0 / 0 / 7 | ssh `2.433e-14` |
| independent | 4 | 0 / 0 / 7 | ssh `2.833e-14` |
| independent | 5 | 0 / 0 / 7 | ssh `2.199e-14` |
| independent | 6 | 0 / 0 / 7 | ssh `4.879e-14` |
| independent | 7 | 0 / 0 / 7 | v `7.720e-14` |
| independent | 8 | 0 / 0 / 7 | u `6.870e-14` |
| independent | 9 | 0 / 0 / 7 | v `1.331e-13` |
| independent | 10 | 0 / 0 / 7 | v `2.285e-13` |
| given entry | 1, 3 | each 5 / 2 / 0 | T `2.361e-16`, `2.360e-16` |
| given entry | 5, 7 | each 5 / 2 / 0 | S `3.524e-16`, `3.523e-16` |
| given entry | 9 | 5 / 2 / 0 | T `2.349e-16` |
| given-entry two-step | 2 | 0 / 3 / 4 | v `2.385e-14` |
| given-entry two-step | 4 | 0 / 0 / 7 | ssh `2.763e-14` |
| given-entry two-step | 6 | 0 / 2 / 5 | ssh `2.165e-14` |
| given-entry two-step | 8 | 0 / 2 / 5 | ssh `1.772e-14` |
| given-entry two-step | 10 | 0 / 3 / 4 | v `1.844e-14` |

Score SHA-256 (first 16): independent kt 1-5 `3d4a3b84a46c41a6`,
independent kt 6-10 `b38606994f8bdd3d`, given entries 1/3/5
`d55c664955176de9`, and given entries 7/9 `f84ff040e064b692`.

**G9 is not certified.**  Therefore this round certifies none of the proposed
ORCA2 fold import: the stage-1 V transport mask, round 46's exchanges,
round 47's FCT fold statements, and round 48's end-of-stage fold and stage
pair all remain pending G9 certification.

## 4. Tests, controls and gates

The new literal-source value test is jitted with traced tracer and metric
transports in x64.  Removing the `nemo_literal_transports` wiring makes its
first-divergence check fail on 766 / 768 values (max relative difference
`1.75e-12`); restoring it passes.  The G9 content test is value-based.

Focused results:

- G9 part 1: 16 passed; stage-fold test: 1 passed; remaining G9 tests:
  3 passed in 266.28 s after the synthetic fixture was supplied the required
  stage transport geometry;
- full shared FCT unit file: 32 passed;
- literal wiring plant: failed as required; the final restored test passed in
  3.81 s;
- G5/G8 jaxpr hashes: unchanged as reported in section 1.

The exact `land.sh` default battery, including the default citation gate,
private-import ratchet, TKE, recipe, freshwater, face-mask, prognostic-state
and ORCA2 parallel-receipt tests, passed **144 / 144** in 1,335.33 s.  The
receipt-specific gate found 4 citations, 0 failures, 0 unmapped citations and
0 failing map entries.  Its planted +2-line shift of the final-write citation
above exited 1 with `SYMBOL-NOT-AT-LINE`.  The required
re-anchor moved 74 map entries and 18 historical audited-receipt pins; four
changed-code anchors then received symbol/extent corrections before the clean
gate.  Transcripts are in `gridladder_rounds/round50/final/`.

## Choices

ASKED: literal shared-FCT form; G9-only selection; kt-1 stage walk; both-label
kt 1..10 score; certification verdict.

UNASKED, offered for revert: the direct literal-source unit regression only.

UNVERIFIED: the owner of the residual 28 T / 17 S after-advection rate cells;
the owner of the 613 / 248 pre-implicit-content cells; the first kt-2 DEBT
statement.  No G9 or ORCA2 fold-import certification is claimed.

COMMITS_IN: /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/gridladder_r50-872514284
ACQUISITION_NEEDED: NONE
DECISION_NEEDED: NONE
ROUND_STATUS: LANDED — literal FCT is selected and at-bar at kt 1; G9 remains uncertified because kt 2 has DEBT rows.
