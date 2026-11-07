# Preregistration: NEMO-testcases L2 GYRE round 66

Date: 2026-09-12. Frozen before the operand-substitution measurement and
before any production change.

## Immutable evidence and active statements

The admitted oracle is
`round64/oracle_krhs_split/oracle_krhs_split_kt00000002.bin`, produced by
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`.  Admission must reproduce
PASS with 43/63 exact inherited records, 20 changed records, and 132 admitted
consumed-field values.  Its seven independent content/TKE calibration rows
must remain exactly zero unequal; stamp, wrong-commit, truncation, and one-ULP
plants must exit nonzero.

The compiled stage-3 arm sets `rDt=rn_Dt` at
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:242-246`, calls
`tra_zdf` with `Kbb=N`, `Kmm=N+1/2`, and `Kaa=N+1` at `:917-965`, and
`tra_zdf` passes that full-stage `rDt` as `p2dt` at `trazdf.f90:150`.
`tra_zdf_imp` writes
`e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` at `trazdf.f90:548-562`, with
`e3t(Kxx)=e3t_3d*(1+r3t(Kxx)*tmask)`.

legoESM's active shared WS-RK3 statement is
`ocean_model_latlon_cgrid.py:2018-2024`: it starts the final content from
`h_k_old*tr`, subtracts the complete stage-3 FCT content divergence, then
adds `dt*h_one_half*stage_source_rates[2]`.  The production call supplies the
step-entry tracer and thickness at `:7639-7667` and the already-computed
stage-2 tracer through `resume=(2, ...)` at `:7671-7677`.

## Operand prediction and exact falsifier

Candidates, scored independently for T and S, are: tracer `Kmm` versus `Kbb`
in the left product; tracer-term thickness `Kmm/Kaa` versus `Kbb`; tendency
thickness `Kbb` versus `Kmm`; `p2dt` full `rn_Dt` versus a stage fraction; and
the `r3t` stretch family behind each thickness.

**Prediction: the tracer level in the left product is the owner: legoESM's
effective live operand is the stage-2 `Kmm` tracer where NEMO reads `Kbb`.**
The magnitude discriminator is the already-recorded `Kmm-Kbb` tracer change:
when multiplied by layer thickness it can be O(1e-3) content, unlike the
O(1e-6) floor implied by the measured FCT Krhs debt.

CONFIRM only if replacing that one live tracer operand by NEMO `T/S(Kbb)` in
legoESM's own content association reduces the T maximum from about
`1.679e-3` to at most `2e-6` content units, while each other one-operand arm
does not; and the reverse arm, replacing only NEMO's `Kbb` tracer by the live
operand, reproduces the removed O(1e-3) difference.  REFUTE otherwise.  If no
single reciprocal substitution meets both tests, report coupled/unresolved
and land no physics change.

For both directions, hold all unselected operands fixed and score exact
unequal count, RMS, and maximum absolute difference on the 18,000 wet cells.
The baseline reconstruction must reproduce the statement it claims to rebuild.
A one-ULP wet-cell operand plant must make the exact census nonzero.  Arrays,
not printed summaries, are compared in fp64.

## Landing and downstream gates

Only a confirmed shared-statement transcription may land.  Preregister GYRE
kt=3 T/S after landing at the Krhs-debt scale (temperature approximately
`1e-6 K`, with salt reported independently); REFUTE the trajectory benefit if
kt=3 T falls by less than 5x.  Score GYRE kt=1..10 and days 1..30 against the
unchanged decision-36 before arm, register every moved row, and require the
first-over-bar boundary not to move earlier.

LOCK_EXCHANGE and OVERFLOW must be resolved from their cards and their rows
through the same stage-3 content statement measured.  DINO's separate
leapfrog content statement is not changed by an RK3-only transcription; any
shared helper it consumes is measured explicitly.  ORCA2 may remain
UNMEASURED only with an exact executable specification on the final commit.

Only after this walk, localise the remaining approximately `6e-11` FCT debt
from the existing round-46 operands.  If its handed-in stage-2 velocity or
transport operands are already unequal, assign it to the momentum ladder and
stop.  No FCT WRITE instrument is permitted unless the existing record cannot
name the first unequal statement.

No configuration choice, carried-state change, NEMO source modification,
NEMO build/run, year-harness change, reconciliation-gate change, freshwater
pair change, or #1484-guard change is authorised.
