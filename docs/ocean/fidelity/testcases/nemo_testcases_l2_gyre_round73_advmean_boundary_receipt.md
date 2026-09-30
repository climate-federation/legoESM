# NEMO-testcases L2 GYRE round 73: external transport-mean boundary receipt

Date: 2026-09-12. Final disposition: **STOPPED FOR RECORD; no production
physics change landed**. The admitted oracle producer is
`7be44bb51402256ca8d77f734867a0c9d651bb9b`; its 67-file comparison has 47
exact files, 20 changed files, and 132 individually admitted differences.

## Verdict and first non-bit statement

The preregistered U-face prediction is **CONFIRMED**: `un_adv` is the first
non-bit input. All 580 wet U cells differ, with maximum absolute difference
`0.00012029895814569258` against a NEMO maximum magnitude of
`2.7221232414391623`. The V analogue is also already non-bit: all 570 wet
`vn_adv` cells differ, with maximum `0.00010099463563850719` against a NEMO
maximum magnitude of `2.4183986570658718`.

This is the first non-bit statement in compiled execution order. The active
GYRE `np_HYB` branch consumes `un_adv` and `vn_adv` while forming `zub` and
`zvb` at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:287-291`, before it
forms `zFu` at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:295` and `zFv` at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:296`. Therefore no
inverse-depth, barotropic-velocity, thickness, mask, or transport association
downstream of those inputs owns this walk. Those rows are **UNREACHED**, not
inferred.

The clean report is `round73/round73_stage1_transport_clean.json`, stamped to
clean measurement commit `cecf9da77c9d1791787d27074e9f5731f77e0fb8`, with
SHA-256 `934f13b8edbdd8116217435e87984f3b7fea152676d2df30818af2f26aab4989`.
All record, live transport-average, geometry, and state arrays are float64.
The independently reconstructed ordinary state has 23 leaves and 198,956
cells, with zero unequal cells and zero maximum difference.

The measured input discrepancy exactly reproduces round 72's downstream
transport census: `zFu` differs in all 17,400 wet cells with maximum
`0.8916110997497526` and RMS `0.15292562665127313`; `zFv` differs in all
17,100 wet cells with maximum `0.7485346468365606` and RMS
`0.12183491024017724`. The record's own `zub`, `zvb`, `zFu`, and `zFv`
associations replay bit-for-bit, so the stop is upstream rather than a parser
or association failure.

## Admission and controls

The acquired round-72 record was re-admitted before measurement. The clean
admission report is `round73/round73_admission.json`, SHA-256
`5e3041db0e175666b67b7564bc07b9d1ec8d9ff92275033bcebf9e9e687f9019`, and
reports `47/67` exact, 20 changed, and 132 admitted values. The independent
record gate remains **AT-BAR**; its report is
`round73/round72_record_gate.json`, SHA-256
`c165a24cd79f7343db011307ea501312fa0eab12097eaae382b2e96361824228`.

Both controls execute and exit nonzero:

- The record-replay ULP plant changes one `zFu` value by one `nextafter` ULP
  and the exact gate fails with one unequal cell and maximum
  `7.275957614183426e-12`. Its log SHA-256 is
  `2f04abcb34ea0e12bb5ae48f8082259d6a737fca1c755d1153d0523f8fdbfad5`.

- The null-`un_adv` plant replaces the oracle target with the live value. It
  changes the first-boundary verdict to **REFUTED**, zero unequal cells, and
  zero maximum difference, then exits nonzero. Its report is
  `round73/round73_null_un_adv_plant.json`, SHA-256
  `217ae44ef6c3da42cb41af881422e8e0be3b9cbd7b18e958e111cb0e5dc59acf`.

Dry-cell division warnings are inherited diagnostics; every reported
comparison is restricted to its owned wet mask. No post-hoc causal arm was
promoted.

## Acquisition prepared for the upstream owner

The required kt=2 external transport-mean record is prepared at
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round73_advmean/run.sh`.
It creates `GYRE_OMIP_L2_P3_SM_R73ADV2` from the admitted R72 configuration,
copies EXP00 and MY_SRC file by file, and refuses existing targets. The source
card only opens the already-defined advective-mean stream at kt=2 and widens
its four WRITE-only predicates; it does not alter a numerical statement.

The source of the missing input is now mechanically specified. At every one
of 50 external substeps, compiled NEMO reads `wgtbtp2`, `zhU`/`zhV`, and the
metric reciprocal, then performs the accumulator statements at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:559-572`. It divides
the completed sums by `r1_wgt2s`, records the pre-LBC result, applies the U/V
boundary exchange, and records the post-LBC handoff at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:797-817`.

The acquisition gate requires the exact 3,789,976-byte record, exact header
and commit/digest stamp, all 100 U/V entry-plus-increment associations, and
both normalization associations bit-for-bit. Stamp, header, truncation, and
replay-ULP plants must each exit nonzero. Admission allows only the named new
record and requires the final restart and mesh mask bit-identical. The exact
source patch dry-applies and its preprocessed result passes
`gfortran -fsyntax-only` with the R72 build includes. Per campaign prohibition,
this round did not invoke makenemo or mpirun.

## Rule 12 card

| card | changed statement | disposition |
|---|---|---|
| GYRE | none in production; measurement code and a WRITE-only acquisition card only | **STOPPED FOR RECORD** at the already-non-bit `un_adv` input. The kt1--10 ladder and days 1--30 are **UNREACHED**. No registered or AT-BAR row moved, and first-over-bar cannot move. |
| LOCK_EXCHANGE | none | No candidate production statement exists to execute; tank gating is **UNREACHED** and the shared implementation is unchanged. |
| OVERFLOW | none | No candidate production statement exists to execute; tank gating is **UNREACHED** and the shared implementation is unchanged. |
| DINO | none | **UNREACHED**. The external-mode transport accumulator is shared-statement risk; any future edit requires DINO's own row gate and cancelling-pair analysis before landing. |
| ORCA2 | none | **UNMEASURED WITH SPEC**: resolve its compiled external-mode card; record every substep's entry accumulator, weight, `zhU`/`zhV`, metric reciprocal, exit accumulator, divisor, pre-LBC and post-LBC values for kt1--10; replay in compiled association; register every moved row, retain all AT-BAR rows, and forbid an earlier first-over-bar. |

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest changed.

## Review and focused checks

The required independent pass was invoked twice with `codex exec --sandbox
read-only` against the complete committed round diff and an adversarial prompt
covering the first boundary, plants, compiled citations, acquisition safety,
and every Rule 12 row. The retry also used its ephemeral mode and ignored user
configuration. Both invocations exited 1 before reviewing because the client
could not initialize in this filesystem sandbox. Its terminal result, quoted
verbatim, is: **“Error: failed to initialize in-process app-server client:
Read-only file system (os error 30)”**. There is no SHIP/DO NOT SHIP verdict;
the independent review is **UNMET/BLOCKED**, and absence of a verdict is not
approval. The identical first-pass and retry logs have SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No production numerical diff is being landed under that blocked review.

The operator's isolated citation-regression request was run first on the
fresh branch tip: the complete citation test file reported 16 passed. The
line shifts had already been corrected by ancestor commit `3d6dfd3015d3`, so
there was no remaining defect and no no-op fix was manufactured.

The focused run reports 49 passed in 1.79 seconds; its JUnit artifact has
SHA-256 `187fe127c63af72a5651800d2874d3d33da6e0fc66c42f6d274d62dbaad5fd74`.
It covers the inherited record gate, round-73 measurement and acquisition
plants, time-level registry, and the complete citation test file. Shell parse,
Python compilation, new-file Ruff checks, patch dry-application, Fortran
syntax proof, and `git diff --check` also pass. The receipt-citation gate
audits every source claim above against the R72 producer's compiled branch;
its shifted-citation plant targets the active `np_HYB` range and exits
nonzero with SYMBOL-NOT-AT-LINE.

## ASKED / UNASKED and OPEN

| state | item | disposition |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | configuration, carried state, stabilizer, NEMO, or harness change | none performed |

OPEN for round 74: an operator must run the prepared round-73 acquisition and
admit its outputs. Then read and cite the new target's compiled writer. Walk
the kt=2 U accumulator in actual substep order: entry `un_adv`, `wgtbtp2`,
`zhU`, `r1_e2u`, the left-associated increment, and exit `un_adv`, stopping at
the first non-bit input or association. Continue through all 50 substeps only
while exact, then check normalization and pre/post-LBC boundaries. Walk V only
after the U owner is named. Do not infer from kt=1, do not resume the held
round-70 output-pair patch, and do not edit production until an independently
computable shared legoESM statement is bit-exact on NEMO inputs and has a
preregistered full Rule-12 causal card.
