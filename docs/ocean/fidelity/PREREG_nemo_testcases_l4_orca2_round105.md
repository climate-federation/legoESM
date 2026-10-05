# Preregistration — ORCA2 round 105 EEN accumulator acquisition repair

Date: 2026-10-02. Base: `fe48ccb3197d9bfcd027ede0edbf62a2d66ee5b7`.
Scope is ocean-only instrumentation. Every later numerical result will be
labelled **independent** because hierarchy rung 0 starts from NEMO's own
from-rest state. Sea ice, every card, configuration field, threshold,
stabilizer, carried state, and the ORCA2 `unmeasured_features` tuple remain
unchanged.

## Observed failure and compiled path

The operator's round-104 run wrote one complete operand file on each rank at
kt=1, then NEMO aborted at kt=2 with `round104: cannot open EEN operand record`.
The writer constructs every filename with the constant `nit000` and opens it
with `STATUS='NEW'` (`l4_r104_een_accum.F90:23-27`). The executing compiled
path calls `dyn_cor_2D_init` at kt=1 and again whenever `lk_linssh` is false
(`ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:302`), and the dump is
inside that routine (`:1294-1297`). Thus the second call tries to create the
already-existing kt=1 filename. This is an instrument defect, not physics.

The repair uses a fresh target. A module initialization routine obtains an
absolute output directory from the launch environment, opens exactly one
rank-tagged file with `STATUS='NEW'` during `dyn_spg_ts_init`, and retains its
unit. The first EEN coefficient construction writes and closes it; later calls
return without opening another file. The launcher creates the directory before
MPI starts and exports its absolute path. No oracle calculation is changed.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R105-P1 | The round-104 abort is solely the repeated open of the constant `nit000` filename. | The fresh run prints one initialization and one dump per rank, reaches `STOP 0`, and creates exactly two rank-tagged operand files. | Any open failure, second dump, missing rank, or nonzero NEMO exit: **REFUTED**; retain the failed evidence and repair only the newly named instrument defect. |
| R105-P2 | The repaired acquisition is observational only. | All 20 terminal restart shards are byte-identical to the admitted round-98 source run. | Any byte differs: refuse the record; no numerical result is citable. |
| R105-P3 | The self-describing record is complete and internally exact. | Both ranks cover the global domain exactly once; all 16 fields parse from their headers; every final coefficient equals `scale * accumulator` bit-for-bit. | Missing rank/cell/field or any reconstruction mismatch: refuse the record. |
| R105-P4 | The instrument and admission controls bind. | Syntax/layout proof passes; path, duplicate-open, header, name, dimensions, truncation, missing-field, signed-zero, swapped-rank, restart-byte, source-layout, and producer-content plants all fire. | Any plant stays green: the acquisition is invalid and no result is citable. |
| R105-P5 | Once admitted, the first remaining non-bit coefficient operand is in NEMO's vertical accumulation, not its final scale. | At every non-fold final signed-zero mismatch, the paired scale is bit-exact and the accumulator is the first unequal operand. | Any scale differs first: **REFUTED**; walk that exact compiled scale statement before changing production. |

## Landing bar

This round may request the record after R105-P1 through P4 are mechanically
encoded and their pre-run plants fire. A production statement can land only
after the record exists, R105-P1 through P5 are measured, all eight coefficients
are bit-exact given NEMO operands, no ORCA2 AT-BAR row leaves the bar, the first
over-bar checkpoint is not earlier, and the registered GYRE, DINO, tank,
citation, focused-test, ocean-fidelity, and review gates pass. Without the
operator-run record the round is `STOPPED_FOR_RECORD`.

