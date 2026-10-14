# ORCA2 round 277 preregistration — ascending-sum rung order and the G3 stored ratio

Frozen before any round-277 ladder runs.  Protocol, row classifier and
labels are round 276's (`nemo_testcase_l4_orca2_round276_omt_source_census_gate.py`),
driven by `nemo_testcase_l4_orca2_round277_rung_census_gate.py`.  Counts are
RMS directions `toward / away / score-equal`; early kt=1..3, late kt>=4;
labels never mixed.

## Existing cards between OMT-4 and shipped rung 10

The only legoESM cards on that span are OMT-4, rung 0 (= OMT-5) and the
shipped rung-10 card.  NEMO records exist for hierarchy rungs 1..9 but no
legoESM card does; none is built this round.  NEMO's resolved
`output.namelist.dyn` diff, already read (it is a deck read, not a
measurement): OMT-4 -> rung 0 is one physics switch (`ln_traldf_off` ->
`ln_traldf_lap`) plus restart protocol; rung 0 -> rung 10 is 19 physics
switches over ten hierarchy modules plus 339 rows of newly-read blocks.

## Predictions and falsifiers

- **R277-P1.** Re-run at this tree, the ascending-sum census reproduces
  round 276 exactly: OMT-4 103/80/12, rung 0 91/86/18 (independent), rung 10
  78/92/10 (given NEMO's entry).  Falsifier: any count differs (the package
  diff from round 276's tree is one default-off selector).
- **R277-P2.** Rung 10 under the independent label (its own entry SSH, the
  label rung 0 carries) is also majority-away.  Falsifier: net-toward, which
  would put the flip in the entry-SSH bridge, not in the ten modules.
- **R277-P3.** The first rung where the sum turns away is rung 10, whose
  predecessor rung 0 differs by more than one switch, so the round stops for
  a decision on which switch to isolate.  Falsifier: OMT-4 or rung 0 already
  away.
- **R277-P4.** G3's stored stage ratio `r3u(Kmm)` (dynzdf.f90:192-198;
  stprk3_stg.f90:215-217) has no host without the G2 divisor statement it is
  an operand of, so it is measured as the pair {divisors, stored ratio}
  against the control, and as an increment against divisors alone.  The pair
  fails Decision 96 on OMT-1 (the divisors alone were 61/78 there in round
  276).  Falsifier: the pair passes D96 on rung 0, OMT-1..4 and rung 10; it
  then lands as one cited unit.
- **R277-P5.** Selecting the stored ratio without its host refuses (plant
  fires); an arm can never be a silent no-op.

No configuration, stabiliser, carried-state or sea-ice change.
