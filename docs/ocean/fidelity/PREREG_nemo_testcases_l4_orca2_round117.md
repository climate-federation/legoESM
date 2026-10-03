# Preregistration — ORCA2 round 117 remaining U EEN recurrence walk

Date: 2026-10-03. Base: `a664ca4243bf0a2817e9bb321672fa07ad19574a`.
All ocean numbers in this round are **independent**: hierarchy rung 0 starts
from NEMO's own from-rest state. No model physics, card field, configuration
value, carried state, threshold, stabilizer, sea-ice selector, or
`unmeasured_features` entry may change before the round-116 record admits and
the three recurrences have been walked in compiled source order.

## Admitted boundary

Round 115 made the northwest-U stored product bit-exact and named the first
non-bit statement as the carried EEN recurrence addition. Ordinary host IEEE
zero addition closed 1,618 accumulator-before and 5,197 accumulator-after
signed-zero bits without moving a nonzero value. Round 116 then committed an
additions-only recorder for the immediately following northeast, southwest,
and southeast U recurrences. The operator reports that acquisition completed;
this round treats that report only as notice that a record may exist, not as
admission or a measurement.

The compiled rung-0 source evaluates northeast, southwest, and southeast U in
that order after northwest U
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1269-1273`).

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R117-P1 | The round-116 additions-only instrument is passive and complete. | The admission gate reports exact two-rank coverage, bit-exact kt=1..10 restarts and inherited streams, exact product/recurrence replay, exact terminal accumulators, and every plant fires. | Any refusal rejects the record; repair the instrument and request a new target before quoting recurrence values. |
| R117-P2 | Northeast U is the next non-bit source boundary and differs only in exact-zero signs in the carried recurrence. | Every NE operand and stored product is bit-exact; the first difference is `before_ne` or `after_ne`, all unequal values compare numerically equal to zero. | The first earlier unequal field owns the walk; any nonzero movement refutes the IEEE-zero arm. |
| R117-P3 | The source-faithful IEEE-zero addition arm closes every NE signed-zero recurrence bit without moving a nonzero value. | NE before/after become bit-exact, candidate-versus-baseline movement is signed-zero-only, and the planted candidate bit refuses. | Any remaining bit or any magnitude movement rejects the arm. |
| R117-P4 | Southwest then southeast U have the same first boundary and close under the same single arithmetic statement. | Each path is walked independently in compiled order; operands/products are exact and before/after become bit-exact with zero magnitude movement. | The first failed path owns the walk; retain all earlier results but do not generalize or land. |
| R117-P5 | No production statement lands from the U-only record. | The round ends HELD or requests the missing four V recurrence streams unless an already-admitted, equally controlled V record proves the same statement on every V path. | Landing without all eight coefficient paths and the full trajectory gates is a process failure. |

## Measurement and landing bar

First run the committed round-116 admission checker against the operator's
existing target and require all runtime plants to fire. Only then may a
committed round-117 walk compare each recorded recurrence with legoESM's
production-JIT CPU fp64/libm replay. It must print shapes and dtypes, score bits
and magnitudes separately, and retain the first unequal location and exact
64-bit values.

The controlled arm changes only the sign rule for adding two exact zeros:
the result is negative only when both operands are negative, matching the
ordinary IEEE addition used by the compiled NEMO statement. No tolerance,
stabilizer, configuration change, or state change is allowed. A production
landing additionally requires all four V paths and the complete
ORCA2/GYRE/DINO/tank/generic-card gates; otherwise this round is measurement
and acquisition only.

## Frozen addendum after the first falsifier

The first committed walk refuted R117-P4 before recurrence arithmetic:
southwest and southeast U each have 68 magnitude-unequal neighboring `vmask`
values at global row `j=0`, level `k=0`; northeast U closes exactly under the
registered zero-addition arm. No statement after the mask has been interpreted.

The compiled initialization forms `vmask` and then applies its ordinary
V-grid lateral boundary condition
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dommsk.f90:211-232`). With no
southern MPI neighbor or meridional self-periodicity, the compiled boundary
dispatcher selects constant fill, whose default land value is zero
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:1815-1819,1852-1872,2129-2136`).
The executed EEN statements then consume `vmask(ji,jj-1,jk)` and
`vmask(ji+1,jj-1,jk)` for southwest and southeast U
(`ORCA2_OMIP_L4_R110EENFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1271-1273`).

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R117-P6 | Replacing only the cyclic southern neighboring-V mask association with NEMO's constant-zero fill closes the 68 SW and 68 SE mask/product magnitude differences. | Mask and stored product become bit-exact on both paths; NE and every non-southern value remain unchanged. | Any remaining bit or any movement outside those 136 registered values rejects the arm. |
| R117-P7 | With P6 applied, the already-registered IEEE-zero addition closes the remaining SW/SE recurrence sign bits without magnitude movement. | All eight recorded fields become bit-exact for NE, SW, and SE; every candidate movement beyond the 136 mask/product values is signed-zero-only. | The first remaining item owns the walk; no production change lands. |
| R117-P8 | The next missing coverage is the four V recurrences, not another U operand. | All four U paths are exact and no equally controlled V recurrence stream exists. | An admitted V stream must be used if found; otherwise write a rank-complete additions-only acquisition. |
