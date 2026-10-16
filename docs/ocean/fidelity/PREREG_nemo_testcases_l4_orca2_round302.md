# ORCA2 round 302 preregistration — per-kt, per-region history and the kt-2 departure

Date: 2026-10-16. Frozen base: `08d30232a`. Scope: OMT-4 (round 223's card,
the card round 298-301 ran), record root round 222's
`orca2_omt4_frames_10step_a_np2` (stage frames `s0..s3`, kt 1-10, both
ranks, fields T, S, u, v, ssh). No production package, card, deck, selector,
carried state, sea ice, `unmeasured_features`, runoff depth (D126) or
after-SSH selection (D132) changes in this round.

## Definitions (frozen)

- Regions: **fold band** = the last three T rows (j = 145-147 of 148, as
  rounds 298-301), **interior** = rows 0-144.  Support = NEMO's own wet mask
  from the rank mesh files (umask / vmask / tmask; ssh on tmask level 0).
- Per field, region, kt and stage: unequal count (D127 subnormal rule),
  max |card - NEMO|, and **relative RMS** = RMS(card - NEMO) / RMS(NEMO),
  both over the region's support.  Over the bar = max |card - NEMO| /
  max |NEMO| > 1e-15 on the region.
- Runs (all free runs from the card's own initial state, which must be
  bit-identical to NEMO's kt-1 entry; that identity makes the "given NEMO
  entry" label the same run):
  - **card** = the OMT-4 card as built, no private hooks.
  - **unit** = rounds 298-301's run: round 249's private atomic fold unit
    hooks plus round 245's barotropic surface-mask arm (`v_only`,
    `t_pivot_north_neighbor`).  NOTE found while reading, before measuring:
    rounds 298-301 scored this run, not the shipped card.
  - **carried** (SCRATCH, D132 evidence, not landed) = card with
    `nemo_first_wzv_after_ssh='rk3_extrapolated_carried'`.

## Frozen predictions and falsifiers

1. **R302-P1 — the review numbers reproduce.** On the **unit** run the
   step-entry U relative RMS is fold 12% / 59% / 281% at kt 2 / 3 / 6 and
   interior 1.7% / 2.7% / 4.8%.  Falsifier: any of the six differs from the
   quoted value by more than 10% of itself.
2. **R302-P2 — the shipped card has the same early fold excess.** On the
   **card** run, kt-2 entry U fold relative RMS exceeds the interior's.
   Falsifier: fold <= interior at kt 2.
3. **R302-P3 — the kt-2 departure is made during kt 1.**  On the card run
   all five fields are exact at kt-1 stage 0, and the first over-the-bar
   stage frame is kt 1 stage 1 for U (both regions).  Falsifier: kt-1
   stage-1 U exact in either region.
4. **R302-P4 — the carried slot does not touch kt 1 and leaves kt 2
   unchanged.**  carried vs card: kt-1 stages bit-identical (the slot holds
   the entry height on the first step); kt-2 and kt-3 relative RMS of U, V
   and ssh move by less than 10% of themselves in both regions
   ("unchanged").  Falsifiers: any kt-1 stage difference; a kt-2 reduction
   of 10-90% ("shrinks") or more than 90% ("collapses").
5. **R302-P5 — no landing.**  The first non-bit statement inside the first
   departing stage is not a transcription defect separable from the held
   fold unit or D126/D132 with the admitted records.  Falsifier: a single
   cited statement that, applied to the card, makes that stage exact in the
   departing region.
