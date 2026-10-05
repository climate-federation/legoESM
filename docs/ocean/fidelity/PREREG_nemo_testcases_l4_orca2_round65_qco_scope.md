# ORCA2 round 65 preregistration — QCO boundary scope

Date frozen: 2026-09-28

Base: `a18927ae417f4606a1b1a951529388bff79d7ae0`

Claim labels: compiled routing is **independent** of initial state; the
record-backed tracer rows are **given NEMO's recorded entry**.  These labels
are not mixed in one numerical table.  Sea ice, its six selectors, and the
ORCA2 card's `unmeasured_features` tuple remain frozen.

## Source-first scope

Round 64 discharged both children of ORCA2's vector-invariant momentum
advection.  Round 60's “QCO mixed boundary” was measured on OVERFLOW's
flux-form program.  The compiled ORCA2 stage program takes the
`ln_dynadv_vec` arm and evaluates the velocity update at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:467-470`;
it cannot evaluate the thickness-weighted velocity arm at lines 472-480 that
round 60 measured.

The same compiled routine subsequently evaluates the tracer QCO/RK assignment
for every momentum-advection regime at
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/stprk3_stg.f90:670-681`.
Round 48 measured that assignment as non-bit in production and bit-exact under
the source-ordered replay, but its candidate was correctly held because five
later OVERFLOW U rows violated the frozen tank non-regression bar.  This round
reclassifies the boundary; it does not retry or weaken that landing gate.

Repository search found and will reuse:

- `nemo_testcase_up3_card_scope.py` for the instantiated four-card momentum
  regime census;
- `nemo_testcase_l4_orca2_round45_qco_rk_gate.py` for the admitted ORCA2
  record, literal replay, source-ordered replay, fused control, and one-ULP
  plant;
- `rk3_stage_velocity_update` and its existing direct tests for the two
  compiled momentum-update arms.

No duplicate numerical replay will be written.  A small composition gate may
only consume these existing results, assert the compiled branch mapping, and
name the first actual ORCA2 non-bit statement at this boundary.

## Frozen predictions and falsifiers

1. **Resolved execution.** The four-card census predicts zero disagreements:
   ORCA2 and GYRE select vector-invariant momentum; OVERFLOW and LOCK_EXCHANGE
   select flux-form UP3.  Any disagreement stops the round.
2. **Round-60 scope.** Given prediction 1 and the compiled `IF`, ORCA2 predicts
   `vector velocity update = EXECUTED` and
   `thickness-weighted velocity update = NOT_EXECUTED`.  A source citation
   failure, changed predicate, or card disagreement refutes this classification.
3. **First actual ORCA2 non-bit statement.** On the admitted round-45 record,
   current production predicts `57,141 / 228,641` unequal stage-1 T cells and
   `57,169 / 228,641` unequal stage-1 S cells.  The literal and source-ordered
   replays predict `0 / 228,641` unequal for both.  The fused control retains
   the production unequal counts.  Any changed count or an inert control
   refutes the prediction and must be reconciled before attribution.
4. **Plant.** A one-ULP mutation in the literal T replay must add exactly one
   unequal active cell and make the scorer refuse.
5. **Disposition.** If predictions 1-4 hold, the first actual ORCA2 non-bit
   statement at the returned boundary is the shared tracer QCO/RK assignment,
   not round 60's non-executing thickness-weighted momentum assignment.  No
   model statement lands because round 48's independent OVERFLOW refusal still
   controls the candidate.  If any premise fails, the round is held without a
   physics edit.

Failed predictions remain **REFUTED** in the receipt.  Exact cell counts and
bitwise status remain separate from numerical maxima.

## Required validation and OPEN

Run both source instruments and their plants, the composition gate, focused
tests, the 170-test card battery, the single permitted `tests/ocean/fidelity
-n 12` battery, both the default and round citation gates plus a shifted-line
plant, and a separate `codex exec --sandbox read-only` review.  Since no model
file is expected to change, trajectory gates are not applicable unless that
expectation is refuted.

If confirmed, the next round executes Decision 52's independent-start ORCA2
ladder; the month-scale ORCA2 magnitude ranking follows it.

ASKED: re-scope the QCO boundary to ORCA2's executing vector program and name
its first actual non-bit statement.

UNASKED: configuration, selector, threshold, carried state, forcing,
stabiliser, sea ice, and NEMO arithmetic changes.
