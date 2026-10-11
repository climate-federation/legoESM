# NEMO SWG level-1 round 6 — whole-step carried state

Date: 2026-10-11. Base: `5b7c5397d`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round6`.

## Preregistration

This section is frozen before any round-6 implementation or measurement. The
operator's independent round-5 review returned SHIP. Round 5 made every
given-entry and chained stage transition bit-identical, while the independent
production ladder first differs at `kt=6` after five exact whole steps.

Frozen predictions and falsifiers:

1. **Complete step-5 carry audit.** Starting only from the SWG card's shipped
   rest state and analytic wind, the production `step()` path will run five
   steps. Its step-5 public after-state and the exact stage-3 output will be
   compared field by field with the admitted `kt=6, stage=1` entry: SSH, U, V,
   before U/V, before/current U/V stresses, `r3t`, `r3u`, `r3v`, and `r3f`.
   The redundant west U and south V storage, masks, and every other SWG state
   leaf will receive an explicit equal/derived/inert disposition. An omitted
   leaf or a comparison made only on the five restart fields **REFUTES** the
   audit.
2. **Boundary prediction.** The step-5 stage-3 output is predicted
   bit-identical to the admitted `kt=6, stage=1` entry on all 13 recorded
   fields, as round 5's chained row already reports. The independently carried
   public SSH/U/V and stress fields are also predicted exact. Any earlier
   unequal field **REFUTES** the proposed boundary and becomes the first
   source-order owner.
3. **F-ratio attribution.** The first reconstructed-entry difference is
   predicted to be `r3f`. NEMO computes the four T-cell numerator with the
   explicitly parenthesized pairwise association and then multiplies by
   `r1_hf_0*r1_e1e2f` at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/domqco.f90:218-229`; stage 3 writes that
   ratio at compiled
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:336-338`, swaps the after slot
   into `Nbb` at `:358-366`, and the next step records it without recomputing
   at `:147-159`. legoESM's stage output selects source-associated evaluation,
   but `_entry_from_state` currently reconstructs `r3f` without that selector.
   A first difference in any other field, or bit-identical reconstructed
   `r3f`, **REFUTES** this attribution.
4. **Landing arm.** If item 3 is confirmed, the production whole-step entry
   reconstruction will use the same shared source-associated QCO F-ratio
   builder already used by the stage boundary. No new numeric implementation,
   state field, default, or configuration selector will be added. The arm is
   confirmed only if the complete carry audit becomes exact and the
   independently initialized ten-step ladder reaches 50/50 bit-identical
   restart rows. Any remaining row **REFUTES** completion and no 30-day
   acquisition will be written.
5. **Control.** A direct production test will compare the reconstructed
   whole-step entry with the actual stage-3 boundary on a nonzero manufactured
   SSH field whose source-associated and unassociated F-ratios differ. Removing
   the selector must make the test fail. A uniform/zero field or a case where
   both associations happen to be equal **REFUTES** the plant.
6. **Rule 12.** The 338/338 given-entry and 338/338 chained rows may not move,
   and no existing independent AT-BAR row may leave the bar. The only shared
   helper behavior change is an explicit SWG caller opt-in; all other callers
   retain their defaults. TSUNAMI, VORTEX, VORTEX_VEC, GYRE, LOCK_EXCHANGE,
   OVERFLOW, DINO, and ORCA2 will be shown not to execute this `swe_rk3`
   boundary or remeasured. Any regression **REFUTES** landing.
7. **Thirty-day stop condition.** Only after the independent ladder is 50/50
   bit-identical will this round add a passive, admitted 30-day NEMO
   acquisition runner under a new target name, derived from the P3 runner and
   shipped deck. It will print a named `REFUSE` line before every nonzero exit,
   record at least daily comparison boundaries, and require its instrumented
   run's ordinary NEMO outputs to be bit-identical to the admitted P3
   reference. No NEMO executable, `makenemo`, or `mpirun` will run here.
8. The receipt-citation gate must map every compiled-source citation, and its
   shifted-citation plant must exit nonzero. Focused stage, ladder, geometry,
   card, differentiability, and control tests must pass on CPU in fp64. A
   green numerical plant, a shifted citation that passes, or a changed
   certified-card hash **REFUTES** landing.

No configuration or carried-state choice is introduced. The proposed arm
preserves the already selected source association at the next production
boundary rather than adding a state leaf.

## Independent review

Independent review run by the operator.

## OPEN

Frozen until measurement.
