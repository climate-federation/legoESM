# NEMO-testcases L2 GYRE round 74: acquisition-preflight repair receipt

Date: 2026-09-12. Final disposition: **STOPPED FOR RECORD; no production
physics or scientific configuration changed**. The missing external-mean
record was not acquired, so round 73's first non-bit boundary remains open.

## Verdict and first non-bit statement

The operator's round-73 acquisition exited 65 before `makenemo` or `mpirun`.
Its preflight required a resolved `nn_baro = 50` row, but the immutable R72
run prints no such row. The compiled program assigns the automatic count to
`nn_e`, then derives `rDt_e` from it at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:1029-1045`. It prints
the resolved external-step seconds, `nn_e`, and automatic/manual branch at the
same compiled source's `:1064-1067`. The R72 output resolves 14,400 s ocean
steps, 288 s external steps, 50 external iterations, and automatic selection.
The failed check was therefore an instrument defect, not a model result.

Round 73's first non-bit statement is unchanged: the active `np_HYB` branch
first consumes `un_adv` while forming `zub` at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/stprk3_stg.f90:287-291`. All 580
wet U cells differ, with maximum absolute difference
`0.00012029895814569258`. No new oracle payload exists to walk its producer.

## Frozen repair and measurements

The corrected acquisition script is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round73_advmean/run.sh`.
It uses the new configuration `GYRE_OMIP_L2_P3_SM_R74ADV2` and new evidence
directory `round74/oracle_advmean_kt2`, while retaining the R72 source card,
round-73 additive WRITE-only patch, exact record layout, record gate, and twin
admission. It still copies EXP00 and MY_SRC file by file, compares the target
`namelist_cfg` byte-for-byte with the source, and stamps the clean commit that
actually runs the full acquisition.

The preregistered clean preflight is **CONFIRMED**. On committed instrumentation
commit `0cd759b031c0`, `--preflight-only` accepted every actual resolved row,
dry-applied the exact additive patch, preprocessed it with the source build's
keys and includes, passed `gfortran -fsyntax-only`, printed
`ROUND74_ADVMEAN_PREFLIGHT_READY`, and exited 0 before the first `makenemo`
statement. Its log is `round74/round74_preflight_with_status.log`, SHA-256
`18d71eb046f1524be8cc3af706c6ff78392408a9ce2689f76fef385c08961e8d`.

The resolved-row plant is non-vacuous and **CONFIRMED**. It first requires the
real `nn_e = 50` row, changes that row to 49 in a temporary copy, proves 49 is
present, and then exercises the same preflight predicate. It refuses on the
missing `nn_e = 50` row and exits 65. Its log is
`round74/round74_resolved_row_plant_with_status.log`, SHA-256
`dc9b92a9e9ad79781069f81a94e7c65abc30132e39d97cf47e1b33a9c690a2f7`.

No configuration/default, carried state, stabilizer, NEMO source/build/run,
year harness, reconciliation gate, freshwater pair, #1484 guard, or held
manifest changed. The failed round-73 check remains recorded rather than
silently replaced.

## Acquisition and next compiled walk

The replacement card is ready for operator execution. In the source program,
each of 50 external substeps assigns the weight and updates the U accumulator
before V at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:559-572`. It then
normalizes, records the pre-boundary values, applies the active U/V exchange,
and records the handoff at the same compiled source's `:797-817`.

A full operator run must emit the exact 3,789,976-byte record; replay all 100
U/V entry-plus-increment statements and both normalization statements
bit-for-bit; keep the final restart and mesh mask identical to R72; admit only
the named record; and make the stamp, header, truncation, replay-ULP, and
consumed-field plants exit nonzero. The full run's commit stamp is fail-closed:
the script refuses any dirty tree and the gate requires the recorded producer
commit to equal the explicitly expected commit.

## Rule 12 card

| card | changed statement | disposition |
|---|---|---|
| GYRE | none in production; acquisition preflight only | **STOPPED FOR RECORD** at the already non-bit `un_adv` input. The kt1--10 ladder and days 1--30 are **UNREACHED**. No registered or AT-BAR row moved, and first-over-bar cannot move. |
| LOCK_EXCHANGE | none | No production statement changed, so the tank gate is **UNREACHED** and the shared implementation is unchanged. |
| OVERFLOW | none | No production statement changed, so the tank gate is **UNREACHED** and the shared implementation is unchanged. |
| DINO | none | **UNREACHED**. The external-mode accumulator remains shared-statement risk, and DINO's 96--98% per-row cancellation requires its own row and pair gates before any future production edit. |
| ORCA2 | none | **UNMEASURED WITH SPEC**: resolve its compiled external-mode card; record every substep entry, weight, transport, reciprocal metric, exit, normalization, and boundary handoff for kt1--10; replay in compiled order; register every moved row; retain every AT-BAR row; and forbid an earlier first-over-bar boundary. |

## Review and focused checks

The required independent Codex review was attempted twice against the complete
committed round diff. The exact requested invocation exited 1 before review;
its terminal result was **“Error: failed to initialize in-process app-server
client: Read-only file system (os error 30)”**. A retry with a writable
ephemeral client home initialized but network access was denied; its terminal
result was **“ERROR: Reconnecting... waiting for network”**. There is no SHIP
or DO NOT SHIP verdict, so review is **UNMET/BLOCKED**, not silently treated as
approval. The logs' SHA-256 values are
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` and
`289662386ea3bc3c5bd229333bebee40b63fd1a961532d4f9db9377e4a3ae9ae`.

The operator-mandated citation regression was the first test run on the fresh
tip and reported 16 passed; ancestor commit `3d6dfd3015d3` had already repaired
the cited-line shifts, so no no-op fix was created.

The receipt citation gate passed from this heading to EOF with zero unmapped
citations at clean commit `0cd759b031c0`; its report SHA-256 is
`e14fbead4ac4d5e4aed6a3528caae62be08286f9da44327289c2bc377301e2d3`.
Shifting the compiled automatic-count citation by two lines produced
`SYMBOL-NOT-AT-LINE` and exited 1; the plant log SHA-256 is
`2f9c55de84e0b011850a469bc2c9bdfbdad7ca0bc8cd2e722c007e192c141728`.
The focused receipt-plus-record-gate run reported 18 passed in 1.53 seconds;
its JUnit SHA-256 is
`646608338554bdd0c1f34e9306e49706fdfccb03b914b6d8123d8d770651b78f`.
Shell parsing, Python compilation, and `git diff --check` pass.

## ASKED / UNASKED and OPEN

| state | item | disposition |
|---|---|---|
| ASKED | new acquisition target after the failed operator attempt | round-74 target and evidence directory prepared exactly as requested |
| UNASKED | scientific configuration, carried state, stabilizer, or scoring choice | none performed |

OPEN for round 75: the operator must run the prepared round-74 acquisition and
admit its output. Then read and cite the new target's compiled writer and walk
the kt=2 U accumulator in actual substep order: entry `un_adv`, `wgtbtp2`,
`zhU`, `r1_e2u`, the left-associated increment, and exit `un_adv`. Stop at the
first non-bit input or association; continue through all 50 substeps only while
exact, then check normalization and pre/post-LBC boundaries. Walk V only after
the U owner is named. Do not infer from kt=1, resume the held round-70 patch, or
edit production until an independently computable shared legoESM statement is
bit-exact on NEMO inputs and has a preregistered full Rule-12 causal card.
