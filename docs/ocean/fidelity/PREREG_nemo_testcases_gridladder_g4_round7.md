# Preregistration — grid ladder G4 round 7 (gridladder_r25), stage-1 tracer divisor

Frozen before any round-25 G4 measurement.  The housekeeping-only public-symbol
promotion is commit `20cbdf5e8`; it changes no arithmetic.

## Fixed program and labels

- Rung: G4, Decisions 120/121/123 exactly as recorded: NEMO's initial T/S,
  SSS restoring on, NEMO-supplied per-step surface fields.
- Oracle record: `gridladder_rounds/round21/oracle_g4_r21`, build
  `ORCA2G1_OMIP_L1_P5`.
- Precision/platform: legoESM fp64 policy, JAX x64, CPU.
- Stage walk: kt = 1, GIVEN-NEMO-ENTRY, compiled order.  Only after stage 1 is
  bit-identical will the kt = 1..10 INDEPENDENT and GIVEN-NEMO-ENTRY ladders be
  scored, including fold/seam/south/coast/open and full/partial-depth splits.
- Bar: the existing ladder's exact 1e-15 normalised bar; no threshold changes.

## First candidate

NEMO first stores the stage-1/2 surface mass-flux divisor as
`r1_rho0 / (e3t_3d(1) * (1 + r3t(Kmm) * tmask))`, then multiplies EMP, the
Kbb tracer, and that stored reciprocal (`ORCA2G1_OMIP_L1_P5/BLD/ppsrc/nemo/trasbc.f90:284-286`).
The same compiled statements run on the full ORCA2 card
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/trasbc.f90:284-286`, Decision 122).

The current G4 path supplies the generic live top thickness to the shared
stage tracer-source code.  The repository already has the literal shared
builder `nemo_qco_live_t_thickness`; no second thickness implementation will be
built.  Candidate: a required, default-off model option selected only by G4
routes the stage-1/2 `tra_sbc` divisor thickness through that builder.  Other
cards and every unrelated stage operand stay unchanged.

## Predictions

1. Before the candidate, round 23 reproduces 58 unequal T cells and 45 unequal
   S cells at stage 1, each no larger than 7.1e-15 absolute, with the other
   scored stage-1 fields bit-identical except signed-zero U faces.
2. On NEMO's own kt-1 entry and surface fields, the literal divisor is
   bit-identical at all wet cells.  The generic divisor differs on a strict
   subset of the 201 bump cells.
3. Selecting the candidate makes stage-1 T and S bit-identical (0 unequal),
   preserves ssh/uu_b/vv_b/v/T/S exact rows, and does not change U except its
   already-recorded signed-zero representation.
4. Once stage 1 is exact, both kt = 1..10 labels reach the existing bar for
   every scored field, region and depth class.  This is a prediction, not a
   certification.

## Falsifiers and controls

- Any stage-1 T or S cell remains unequal after the literal divisor: prediction
  3 is REFUTED; walk the next compiled statement and do not certify G4.
- Any previously bit-identical stage-1 value becomes numerically unequal: hold
  the candidate and name the moved row.
- Deselecting the option must reproduce the recorded non-exact T/S cells; if it
  does not, the control is vacuous and the measurement is invalid.
- The literal-builder unit plant must still differ from the generic expression,
  and moving the selection off G4 or onto another card must fail the card test.
- G4 is CERTIFIED only if both complete ten-step labels pass every field,
  region and depth split.  A global-only pass is insufficient.

