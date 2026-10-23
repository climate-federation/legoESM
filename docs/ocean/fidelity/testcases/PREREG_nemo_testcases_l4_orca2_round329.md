# ORCA2 round 329 preregistration — the rung-10 comparison builds the record's own initial sea ice (DECISION 157)

Frozen 2026-10-23, before the Decision 122 arms, the kt-10 errors and the
gradient check (Claude round).  Disclosed: the kt-1 entry SSH comparison (E1)
was run once while wiring the card, before this freeze; it is re-run as a
committed test.

## Finding that sets this round

- Round 327: the independent label's first over-bar row is kt 1 entry SSH, a
  uniform +0.015479333813968585 m on all 16,433 wet cells.  The record's NEMO
  run builds a one-category, ponds-off initial ice (its `ocean.output`:
  jpl = 1, ln_pnd = F, rn_thres_sst 2, rn_hts_ini 0.2/0.1, rn_hti_ini 2.5/1.0,
  rn_ati_ini 0.997/0.8; "sea level adjusted by -0.108914 m"), while the card
  builds ORCA2's five-category, ponds-on ice (-0.09343505872684969 m).
- NEMO forms the pond area as rn_apd_ini * a_i
  (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:324`),
  zeroes ponds when ln_pnd = F
  (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:343-346`),
  and subtracts the global snow+ice+pond mass from SSH
  (`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/iceistate.f90:442-459`).

## Change (comparison harness only)

- The card's inline initial-ice/SSH block becomes one recipe function taking
  the &namini values; the shipped card calls it with its unchanged ORCA2 deck
  values.  The comparison card calls the same function with the values parsed
  from the record's `ocean.output` (refuses if any is missing, duplicated, or
  the run did not use the analytic initial ice).  No other setting moves.

## Predictions and falsifiers

- S1 shipped cards unchanged: normalized one-step programs identical base vs
  round on rung 0, OMT-0..4 and rung 10 (7/7); rung 10, 3 production steps,
  all leaf-steps bit-identical.  Falsifier: any program or leaf differs.
- E1 the comparison card's kt-1 entry SSH is bit-identical to NEMO's on all
  26,640 cells.  Falsifier: any unequal bit.
- D1 Decision 122 (with DECISION 139's recomputed safeguard) passes on all
  seven pairs: rung 0 and OMT-1..4 arms are the same card (no change); rung 10
  given-entry rows identical across arms; rung 10 independent loses no exact
  row and its first over-bar row moves later than kt 1 entry SSH.
  Falsifier: an exact row lost or an earlier row worse.
- K1 rung 10 independent with the record ice reproduces the given-entry
  label's kt-10 full-domain errors (round 327: SSH 3.73e-5 m, T 7.70e-5 K,
  S 1.22e-5, u 3.05e-3 m/s, v 3.15e-3 m/s), because every entry field is then
  NEMO's.  Falsifier: any kt-10 maximum differs from the given label's
  (reported with the residual, not hidden).
- G1 rung-10 N = 1 gradients through the comparison card: 0 non-finite,
  reverse = forward to <= 1e-12 relative.  Falsifier: any non-finite or a
  larger mismatch.
