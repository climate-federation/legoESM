# Round 54 preregistration — GYRE step-2 tracer decomposition

Date: 2026-09-11. Frozen base: `58bbb84835d1`. CPU, production JIT,
fp64/scalar-libm only. NEMO source and records are read-only. This first
registration fixes the decomposition before any operator score is read; the
operator owner and its falsifier will be appended and committed only after the
decomposition, and before the ablation ladder.

## Executed tracer program

The compiled driver advances stages 1, 2, and 3 at
`GYRE_OMIP_L2_P3_SM_R46KT2/BLD/ppsrc/nemo/stprk3.f90:201-215`. Each stage calls
the tracer program at `stprk3_stg.f90:773-800`. Stages 1/2 zero the tracer RHS,
then execute advection and surface boundary forcing
(`stprk3_stg.f90:825-863`), followed by the concentration-form RK3 update
(`stprk3_stg.f90:886-908`). Stage 3 additionally executes solar penetration,
lateral diffusion, and vertical diffusion (`stprk3_stg.f90:924-958`). The
stage records' Kmm tracer at the next stage therefore names stages 1/2; the
kt=3 entry names stage 3.

## Frozen decomposition

Starting from NEMO's kt=2 entry, run legoESM's own step and compare against the
round-46 stage records and NEMO's kt=3 entry. For temperature and salinity at
each stage, record wet-cell RMS, max absolute error, unequal-cell count, and
area/thickness-weighted signed error. For the final error additionally record:

- every model level's RMS, max, unequal count, and squared-error share;
- north/south halves, west/middle/east thirds, and their intersections, using
  index masks frozen by the GYRE grid rather than post-hoc error contours;
- surface, interior, and bottom wet-cell aggregates, with every cell assigned
  once; and
- location/value of the two largest absolute errors, checked directly against
  raw arrays.

The stage discriminator is preregistered as follows. A first non-bit stage-1
or stage-2 boundary leaves advection/surface forcing live and refutes a sole
stage-3 owner. Bit-exact stages 1/2 with a stage-3 jump confirms only the
stage-3 group. A top-only signature supports surface forcing or solar heating;
a subsurface column-redistribution signature supports vertical diffusion; a
broad flow-following signature supports advection; a subsurface isoneutral
signature supports K33. These are classification rules, not owner claims.

## Instrument controls

The committed gate must reproduce the reconciled final T RMS
`4.1543946279e-4 K` within `5e-15 K`, find 17,999 unequal wet T cells, and
report zero input mismatch. Swapping a stage oracle, perturbing a nonzero wet
tracer value, truncating a record, changing a producer stamp, and replacing an
array by itself must each exit nonzero. Metrics reject NaN and masks with zero
support. Two reported raw-array rows are spot-checked independently.

## Stop/landing rule

After this signature is measured, append exactly one ranked operator and a
numeric falsifier before scoring existing records or ablations. An ablation
quantifies influence but cannot name a statement. If the winning operator is
not resolved to its first non-bit operand/accumulator by existing records,
write a WRITE-only kt<=2 NEMO instrument plus additive `run.sh` copied from
R46KT2 with `makenemo -r ... -n ...`; do not run it and do not change legoESM.
Rule 12 forbids a shared fix without per-card operator proof.

ASKED: all measurements and stop/landing rules above. UNASKED: none. Forbidden:
configuration/default/state choices, NEMO build/run/edit, GPU use, deletion,
threshold relaxation, per-card guards, merge, push, or changes to the TKE
runaway fix, year harness, and reconciliation gate.

## Post-decomposition owner registration (before operator scoring)

Committed decomposition producer: `f983f550f845`; evidence:
`round54/step2_tracer_decomposition_v2.json`. The v2 artifact retracts v1's
two peak rows because v1 took `abs` after filling dry cells with `-inf`; every
aggregate was recomputed. The corrected dry-cell plant is a direct test.

The first ranked operator is **tra_zdf's complete tracer operand/solve chain**,
not yet a particular statement. The observed T RMS grows from `8.374e-13` at
stage 1 to `3.202e-8` at stage 2 and `4.154394627918e-4 K` at stage 3; S grows
from `5.193e-14` to `2.386e-9` and `5.994e-5`. Temperature levels 0, 1, and 2
hold `0.43839054`, `0.47937584`, and `0.08223341` of squared error; all deeper
levels together hold less than `1.9e-7`. The thickness/area-weighted signed
T and S errors are `-6.74e-13 K` and `4.51e-15`, respectively. This
three-level, two-tracer, column-redistribution signature ranks the
preregistered vertical-diffusion class ahead of solar, surface flux, and
stage-3 advection; it does not statement-exonerate those operators. The
earlier v2 auxiliary arm that reported `3.721e-4 K` is RETRACTED by the v3
producer: it seeded kt=2 from kt=2's post-`zdf_phy` record and advanced that
memory twice. Seeding the actual kt=1 memory leaves T RMS at
`4.15439462791858e-4 K` (within `1.2e-17 K` of the ordinary arm), so the
independently produced kt=1 TKE memory is exonerated as the step-2 owner.

Score in this order: (1) round-38's kt=2 NEMO matrix/RHS/solution through the
shared literal tracer solve; (2) current kt=2 model-path pre-ZDF content,
`avt`, K33, coefficients, RHS, and output against the same record; (3)
round-40 slopes/K33; then (4) one-variable operator ablations. The shared
tridiagonal sweep is CONFIRMED only if its NEMO-given matrix/RHS output is not
bit-identical to the recorded solution; bit identity REFUTES the sweep and
promotes the earliest live model-path operand. The tra_zdf chain is REFUTED
as majority owner if substituting all recorded kt=2 ZDF operands fails to
remove the stage-3-created excess above the measured stage-2 residual in both
T and S. An ablation alone cannot satisfy that statement test.

## Adjudication

Evidence `round54/kt2_zdf_score_v4.json` confirms `tra_zdf` as the owner but
stops one boundary short of a source statement. Given NEMO's matrix, content
RHS, and sweep inputs, all three shared operations are bit-exact. Replacing
only the live effective K reduces T/S RMS from `4.15439462791858e-4` /
`5.99391804444020e-5` to `4.22651006572494e-6` / `2.38836182218273e-7`;
replacing every recorded ZDF operand reduces them to `5.99654331381177e-13` /
`9.56273164723948e-13`. The source-defined `avt == rn_evd == 100` partition is
exact to `9.95e-14 m2/s` max, while the stable partition differs by
`7.76353183052508e-3 m2/s` max. Thus `zdfevd` is exonerated and the earliest
recorded non-bit boundary is the kt=2 `zdf_tke` output (`avt_k` RMS
`2.47083026062888e-4 m2/s`). Existing records do not expose its matrix/RHS,
sweep, mixing-length, and coefficient boundaries, so no source statement is
named and Rule 12 forbids a shared fix. Acquire the registered WRITE-only
record with:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round54_tke_operands/run.sh
```
