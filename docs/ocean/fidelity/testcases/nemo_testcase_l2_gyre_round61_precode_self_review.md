# Round 61 pre-code self-review — split the kt=2 TKE input row

Date: 2026-09-11. Frozen parent
`4baf0d012bda99a109a30a5bb761773814c451bc`; governing preregistration is
`nemo_testcase_l2_gyre_round61_preregister.json`. No operand has been scored.

## Source and path audit

Compiled GYRE calls `zdf_sh2` before `zdf_tke` and passes the resulting field
with the carried `avm_k/avt_k` at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfphy.f90:317-337`.
The executing shear branch multiplies Kmm and Kbb face-velocity differences,
divides by the corresponding live face W metrics, applies face masks, and
wet-only averages with the carried viscosity at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdfsh2.f90:83-114`.
The instantiated legoESM card instead prints
`tke_shear_production="squared_centered"`,
`tke_shear_avm_weighting="tpoint"`, and
`tke_shear_metric_source="tpoint_jacobian"`. That makes `sh2` the first-ranked
candidate before measurement.

NEMO constructs `rn2b` from Nbb tracers and copies it to `rn2` before calling
vertical physics at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/stprk3.f90:159-168`.
Inside the closure, `rn2b` feeds Langmuir and inverse Prandtl, while `rn2`
feeds the buoyancy RHS and mixing length at
`GYRE_OMIP_L2_P3_SM_R59TKE/BLD/ppsrc/nemo/zdftke.f90:353-380`,
`:394-433`, and `:627-629`. The previous closure exit produces the entry
`avm/avt/dissl` at `:691-701`; the kt=2 entry `en` is the previous step's
prognostic exit. Geometry and masks enter the same matrix/RHS at `:416-433`.

## Instrument review before extension

Search found one existing reader and production-path walk,
`nemo_testcase_l2_gyre_round54_tke_operands.py`; no second tool is needed.
Its record parser and seven-row NEMO-only calibration are retained. The needed
extension is to advance the production card exactly once, materialize the
actual kt=2 carried operands, score each against the admitted R59 record, and
run both single-operand and cumulative substitutions through the same
production `tke_vertical_mixing` path. Exact uint64 equality is the bar.

The current walk's first 744-cell row is not by itself an operand attribution:
it supplies NEMO's recorded operands while retaining legoESM's downstream
associations. Round 61 must first show that an actual model-operand baseline
reproduces the registered 744 count; disagreement stops attribution and is
reconciled before any result is recorded.

The operand plant will alter one nonzero, consumed wet value by one ULP and
must fail the exact operand assertion. The existing consumed-closure and stamp
plants remain. No production physics, configuration, model state schema,
trajectory harness, reconciliation gate, freshwater pair, or #1484 guard is
eligible before the frozen prediction is scored.

ASKED: preregistration, source/path audit, and this pre-code review. UNASKED:
none. UNVERIFIED: all round-61 operand counts and owner claims.
