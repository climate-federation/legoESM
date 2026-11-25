# ORCA2 round 303 preregistration — hooked run kt 1-10 after the EVD fix, and its earliest over-bar row

Date: 2026-10-16. Frozen base: `710787b0c` (round 302 landed: ORCA2 EVD mixes
tracers only, `nu_conv = 0`). Scope: OMT-4 (round 223's card), record root
round 222's `orca2_omt4_frames_10step_a_np2` (stage frames `s0..s3`,
kt 1-10). No production package, card, deck, selector, carried state, sea
ice, `unmeasured_features`, runoff depth (D126) or after-SSH selection (D132)
changes unless a cited transcription defect qualifies under D96.

## Definitions (frozen, identical to round 302)

- Instrument: round 302's committed history probe
  (`nemo_testcase_l4_orca2_round302_region_history.py`), arm `unit` = the
  **hooked run** (round 249's private atomic fold unit + round 245's
  barotropic surface-mask arm), `--kts 10`, clean committed tree.
- Fold band = last three T rows; interior = rows 0-144; relative = RMS(card -
  NEMO) / RMS(NEMO) on NEMO's wet support; over the bar = max|diff| /
  max|NEMO| > 1e-15.
- Comparison baseline: round 302's `history_unit.json` (same instrument,
  `nu_conv = 100`, kt 1-7).

## Frozen predictions and falsifiers

1. **R303-P1 — kt 1 stages 1-2 unchanged.** EVD acts at stage 3 only, so the
   hooked run's kt-1 stage-1 and stage-2 rows equal round 302's
   `history_unit.json` exactly (all fields, both regions).  Falsifier: any
   kt-1 stage-1/2 row differs.
2. **R303-P2 — the interior improves at kt 2.** kt-2 entry interior U
   relative RMS falls below round 302's 1.72%.  Falsifier: >= 1.72%.
3. **R303-P3 — the fold-band blow-up persists.** Fold-band U relative RMS at
   kt-7 entry exceeds 100% (round 302: 1170%), i.e. the blow-up is not an EVD
   effect.  Falsifier: kt-7 fold U relative RMS <= 100% and no kt in 7-10
   above 100%.
4. **R303-P4 — the earliest over-bar row is unchanged.** The hooked run's
   first over-bar row is kt-1 stage-1 fold-band V (with T), and its V
   departing support is the 1,319-face stage-1 V mask set named by rounds
   258/283.  Falsifier: an earlier row, or a different V support count.
5. **R303-P5 — no landing.** The first non-bit statement of that row is the
   held resolved stage-mask statement (round 286/288, refused on rung 0) or
   inside the private fold unit, not a separable transcription defect.
   Falsifier: a single cited statement outside both that makes the row exact
   and passes D96 on all seven card/label pairs.
