# ORCA2 round 290 preregistration — stage-1 V source-row boundary split

Date: 2026-10-16

Round 289 proved that NEMO's completed-stage V fold association is exact when
given NEMO's source row, while legoESM's live source row is already non-bit on
1,440/5,400 values.  This round splits that source row at the two matched
program points already exposed by the passive production trace: after the raw
vector-form velocity update and after the barotropic correction, both before
the completed-stage `lbc_lnk`.

The compiled ORCA2 record build writes the raw vector-form update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480` and then
applies the barotropic correction at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:521-528` before the
association at `stprk3_stg.f90:812-820`.

Every value is **independent OMT-4, kt=1 stage 1**.  The instrument reuses the
round-289 passive trace's `stage_raw_velocities` and `stage_outputs`, and the
admitted rank-complete round-280 `pre_v` and `cor_v` streams.  No executable
observer, NEMO run, model edit, card change, configuration choice, carried
state change, or landing is authorised.

## Frozen predictions and falsifiers

1. **R290-P1 — admission and passivity.**  The round-280 record re-admits as
   `PASS_R280_OMT4_STAGE1_ADMISSION`; the traced and ordinary completed states
   remain array-identical in every leaf.  Either failure refuses every number.
2. **R290-P2 — raw-update boundary.**  On source row 146, the candidate raw V
   after the vector-form update remains within the campaign floor of NEMO's
   recorded `pre_v`.  A valued error above `2e-10 m/s` names the raw update as
   the first boundary and refutes the predicted correction owner.
3. **R290-P3 — correction boundary.**  The candidate corrected V reproduces
   round 289's source-row census against NEMO `cor_v`: 1,440/5,400 unequal,
   maximum `0.016207369589460653 m/s`, RMS
   `0.0035899206923965866 m/s`.  With P2 green, this names NEMO's barotropic
   correction statement as the first valued boundary.  Any changed census or
   magnitude refutes the calibration and stops attribution.
4. **R290-P4 — matched increment discrimination.**  Candidate and NEMO
   correction increments are scored on the same source row.  Their difference
   must carry a valued maximum and reproduce the corrected-boundary maximum
   when added back at the matched program point.  A null increment difference
   makes the two-boundary attribution internally inconsistent and refuses it.
5. **R290-P5 — non-vacuity.**  Passivity, program-point order, raw-boundary,
   correction-calibration, and one-bit plants must each make the gate refuse.
   A green plant refutes the instrument.

The outcome is measurement-only and **HELD**.  The first valued boundary is
recorded with its compiled citation; its operands are work for a later frozen
round.  Decision 125's top-V exchange and the resolved stage mask remain off,
Decision 126 remains pending and unchanged, and sea ice remains exactly as the
shipped card carries it.

Independent review: operator, as required for this lane.
