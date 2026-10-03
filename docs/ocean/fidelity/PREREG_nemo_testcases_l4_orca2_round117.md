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

