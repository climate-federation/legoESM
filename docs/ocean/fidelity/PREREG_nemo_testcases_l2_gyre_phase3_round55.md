# Round 55 preregistration — GYRE TKE closure and `rn_mxl0`

Date: 2026-09-11. Frozen base: `3a8d94ea1985`. CPU, production JIT,
fp64/scalar-libm only. NEMO source and records are read-only. The round-54
operand acquisition produced no record: its launcher exited 65 before creating
the target because it looked for inherited `nn_pdl=1` only in `namelist_cfg`.

## Acquisition correction, fixed before the replacement record exists

The resolved R41ADVSP/R46 GYRE run prints `nn_pdl=1`, `nn_mxl=3`,
`ln_mxl0=T`, raw `rn_mxl0=0.04`, `nn_etau=0`, `nn_htau=1`, `ln_lc=T`,
`rn_lc=0.15`, and post-initialisation `nn_eice=0` in `ocean.output:568-600`
(R41ADVSP; the R46 source run prints the same block at `:586-618`). The
configuration reads the reference namelist first and the case namelist second
(`GYRE_OMIP_L2_P3_SM_R41ADVSP/BLD/ppsrc/nemo/zdftke.f90:726-739`), so absence
of `nn_pdl` from the case override is not a falsifier. The repaired launcher
must gate the source run's resolved `ocean.output`, then gate the new run's own
resolved output.

The reference brief calls these “zdfphy nn_pdl arms”; the compiled source has
no such arms in `zdfphy`. The live branches are in compiled `zdftke.f90`:
allocation/deallocation at `:187-193`, Richardson/Prandtl construction at
`:386-405`, and final `avt` update at `:690-694`. Compiled `zdfphy.f90:334-351`
selects TKE, receives `avm_k/avt_k`, and copies them to `avm/avt` before EVD at
`:359`. The replacement writer records both the post-Prandtl closure outputs
and the copied pre-EVD fields, plus all operands needed to rebuild them. Its
target build's own `BLD/ppsrc` becomes the citable source after acquisition.

## Frozen hypothesis and falsifier

Leading candidate: NEMO derives `rmxl_min = 1e-6 /
(rn_ediff*SQRT(rn_emin)) = 0.01 m` (`zdftke.f90:810-816`) and, because
`ln_mxl0=T`, overwrites the raw namelist value with `rn_mxl0=rmxl_min`
(`:828-831`). legoESM currently uses `mxl0_min_m=0.04` in the shared surface
anchor. This is the confirmed defect left unlanded by the TKE-runaway receipt.

Calibration: rebuilding NEMO `avm/avt` from NEMO's own recorded operands must
give zero unequal cells on every consumed wet interface. Substitution ladder:
run the model's own closure path, replace one NEMO operand at a time in source
order, and require the first substitution producing zero unequal `avt` cells
to name the statement. Prediction: replacing only the effective surface floor
`0.04 -> rmxl_min=0.01` is that first zero-cell substitution. Falsifier: any
remaining unequal consumed `avt` cell promotes the earliest remaining operand;
the floor is still landed as a cited transcription but is REFUTED as sole kt=2
owner.

## Frozen trajectory program

Before and after differ only in the shared `ln_mxl0` floor semantics. Run the
existing production-path GYRE ladder for kt=1..10 and a 30-day control with
daily snapshots; score against `year_owners/nemo_seed0`. For kt=3, if the floor
is the sole coefficient owner, the max T/S residuals are predicted to become
the round-54 NEMO-effective-K arm values `1.6275031290e-4 K` and
`6.3278533133e-6 g/kg`, down from `8.7413164119e-3 K` and
`1.3568885211e-3 g/kg`. Any larger post-fix kt=3 max refutes that quantitative
prediction. Day-30 T RMS is predicted to improve from its newly measured
same-HEAD baseline; equality or worsening refutes only the trajectory-direction
prediction, not the transcription. A 360-day rerun is not preregistered because
one CPU member costs about 30 minutes and the user made it conditional on being
cheap; the existing before value is reported, not compared to a missing arm.

Rule 12: GYRE executes the shared anchor and is measured. LOCK_EXCHANGE and
OVERFLOW must be verified from their resolved namelists to run constant ZDF,
not TKE. DINO executes the same shared statement but already has
`mxl_min=mxl0_min_m=0.01`, so exact equality of its built pre/post configuration
is required; its leapfrog branch remains separately measured debt, not cleared
by GYRE. ORCA2 is UNMEASURED-with-spec: a future ORCA2 card must compare the
closure output bitwise with its own compiled NEMO operands before discharge.

ASKED: acquisition repair, shared floor transcription, stated GYRE program and
Rule-12 audit. UNASKED: none. No configuration choice is introduced: the raw
namelist value is replaced by the compiled oracle's mandatory derived value.
