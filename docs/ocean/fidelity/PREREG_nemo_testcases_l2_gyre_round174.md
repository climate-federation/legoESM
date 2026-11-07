# Preregistration — round 174, paired developed solve-input ranking

Committed before scoring any Round-173 restart.  The operator reported
`ROUND173_SOLVE_INPUT_PAIR_READY` for the corrected wet-only retry.  This round
admits that record and ranks the two forced-input families.  It does not change
production physics, configuration, carried state, restart schema, or the
immutable Round-163 before arm.

## Compiled statements and comparison

The producing NEMO build reads one `e3t` and one temperature-content frame for
steps 1081--1440 at
`GYRE_OMIP_L2_P3_SM_R172SOLVEPAIR/BLD/ppsrc/nemo/trazdf.f90:136-150`.
The `e3t_wet` arm selects the imported wet-cell diagonal factor at
`:490-504`.  The `content_wet` arm selects the imported temperature
right-hand side at `:583-609`.  Both arms consume the resulting matrix and
content in NEMO's ordered solve at `:619-624`.

The common reference is the admitted Round-125 step-1440 restart.  The
baseline, `e3t_wet`, and `content_wet` step-1440 temperatures are scored with
the same unweighted binary64 RMS over the 18,000 NEMO wet active cells.  The
forced arm with the larger RMS against baseline carries the larger leverage
on the production complete-K/e3w remainder
`1.241262968697578e-03` K and names the next producer walk.  This inverse
forced-input sensitivity ranks families only; it is not a source-exact
landing proof.

## Frozen predictions and falsifiers

1. Admission confirms `STOP 0` for both arms, both required restarts, binary
   and corrected-input identity, baseline identity to Round 125 at steps 1080
   and 1440, and a non-identical step-1440 restart for each directed arm.  Any
   failed condition refuses both scientific rows.
2. Baseline temperature is BIT against Round 125: zero unequal wet cells and
   zero RMS.  Any baseline difference invalidates the common reference.
3. Both directed arms move at least one wet temperature cell and have nonzero
   day-240 T3D RMS.  A zero row is non-discriminating and refused.
4. The `e3t_wet` RMS is larger than the `content_wet` RMS.  Equality or the
   reverse ordering REFUTES the frozen Round-172/173 ordering; the measured
   larger arm is promoted regardless.
5. Neither directed RMS reaches half the complete-K/e3w remainder,
   `6.20631484348789e-04` K.  A row at or above that value REFUTES the earlier
   expectation that neither family alone carries half the remainder.
6. Scaling the selected arm's wet temperature difference from baseline by
   exactly `1 + 2**-20` changes all unequal cells, changes its RMS, prints
   `STATUS PLANT-FIRED`, and exits nonzero.  A zero exit, unchanged metric, or
   movement outside the registered wet difference invalidates the scorer.

The receipt will retain every refuted prediction, state the two RMS values and
fractions of the `1.241262968697578e-03` K remainder, and name exactly one next
producer boundary.  No landing gate, card census, ladder, month, year, DINO,
tank, or ORCA2 trajectory runs because no model implementation changes.
