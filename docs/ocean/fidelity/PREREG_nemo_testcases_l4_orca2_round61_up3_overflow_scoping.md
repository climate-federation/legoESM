# ORCA2 round 61 preregistration — UP3 statement card scoping

Date frozen: 2026-09-28
Base: `1bbf37814553`
Cards named: `ORCA2-zps`, `GYRE-zco`, `OVERFLOW-zps`, `LOCK-zco`
Claim labels: card scoping **by construction**; OVERFLOW trajectory rows
**independent**; NEMO statements **compiled source**

## The question this round settles

Rounds 57-60 held four times on one question: the source-ordered UP3 U T-face
flux statement is reported to close a `282 / 17,000` direct flux error to
`0 / 17,000`, yet under it 31 OVERFLOW rows move and 20 exceed the frozen
strict 2-ULP bar. The round order names four candidate resolutions: (A) the
two cards select different momentum-advection switches, (B) the candidate is
wrong on the tank's partial-step / closed-wall cells, (C) the tank's certified
record came from a different NEMO build, (D) the candidate changes something
upstream that the tank runs.

## Evidence gathered before this freeze (immutable reads, not runs)

These four reads are of files nobody in this round can change, and each is
reproducible by the quoted command. They are recorded here so the round's
disposition is not presented as a post-hoc prediction.

1. `diff` of the two decks' momentum blocks
   (`ORCA2_OMIP_L4/EXP00/namelist_cfg:346` and `:352` versus
   `OVERFLOW_OMIP_L1/EXP00/namelist_cfg:83` and `:89`).
2. `diff` of the shipped `src/OCE/DYN/dynadv_up3.F90` against the instrumented
   `tests/OVERFLOW_OMIP_L1_P3_R56UP3/MY_SRC/dynadv_up3.F90`, plus the deck and
   `cpp_*.fcm` diffs between `OVERFLOW_OMIP_L1` and `OVERFLOW_OMIP_L1_P3_R56UP3`.
3. `git show f38a958055` — the held candidate's only model hunk.
4. The round-47 OVERFLOW reference report's own per-row status census and the
   round-59 comparison's per-row `field_moves`, read through the committed
   `legoesm.ocean.fidelity.ulp_move_gate.certified_rows`.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R61-P1 | Built from their own builders, `ORCA2-zps` and `GYRE-zco` resolve `momentum_advection='vector_invariant'`, and `OVERFLOW-zps` and `LOCK-zco` resolve `momentum_advection='flux_form'` with `momentum_flux_scheme='nemo_up3'`. | All four constructed configs carry those values. | Any card disagrees with its deck: stop and report the card/deck mismatch as the finding. |
| R61-P2 | The held candidate's model hunk is unreachable from the two vector-invariant cards, because its only call site is the `flux_form` arm of the momentum branch in `ocean_pe_latlon_cgrid.py`. | The edited function has exactly one call site and it sits in the `flux_form` arm whose `else` is the vector-invariant ENE/EEN operator. | More than one call site, or a call from a shared helper: hypothesis (D) is live; bisect the hunks against the OVERFLOW ladder. |
| R61-P3 | Every one of the 20 OVERFLOW rows the candidate pushes over the bar is a row the reference report already registers as `DEBT`, and no `AT-BAR` row is lost. | All 20 violating rows have reference status `DEBT`; the gate's own `AT-BAR -> DEBT` rule does not fire. | Any violating row is `AT-BAR` in the reference: the candidate breaks a certified row and hypothesis (B) is the live one. |
| R61-P4 | On each violating row the candidate's worsening is smaller than the row's own pre-existing residual, so the bar refuses a change smaller than the error it is defending. | Every ratio worsening / reference residual is below 1. | Any ratio at or above 1: the candidate dominates the row's error and must be treated as a trajectory defect. |
| R61-P5 | No model statement lands this round. | The final `packages/` diff is empty and the tree is byte-identical to base `1bbf37814553`. | A model edit is present at disposition: remove it or justify it under the complete shared-card gate. |

## Disposition rule, frozen

If R61-P1 and R61-P2 hold, the UP3 statement cannot change ORCA2 under any
gate, so the round's landing predicate ("ORCA2 ladder improves with the direct
flux at 0") is unsatisfiable by this statement and the walk stops. Whether the
statement then lands on the OVERFLOW card is a threshold question about how the
frozen bar is read on `DEBT` rows; that is the operator's to answer and is
raised as one decision, not taken here.
