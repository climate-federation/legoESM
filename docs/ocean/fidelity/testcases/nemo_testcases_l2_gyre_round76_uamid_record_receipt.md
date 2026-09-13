# NEMO-testcases L2 GYRE round 76 receipt: U-midpoint record needed

Date: 2026-09-12. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Compiled statement and verdict

**STOPPED FOR RECORD; NO PRODUCTION PHYSICS LANDED.** The frozen magnitude
prediction is **REFUTED**. The live instrument's first non-bit row is the
substep-1 U metric transport, but only 2 of 580 wet faces differ, with maximum
absolute difference `9.947598300641403e-14`. NEMO forms that transport as
`zhU=e2u*ua_e*zhup2_e` in its admitted target's compiled branch at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:538-544`.
The preregistered prediction was 580 of 580 faces and is therefore refuted.

The two changed `ua_e` operands differ by at most
`2.117582368135751e-22`; `zhup2_e` is bit-exact. More importantly, the ordinary
production trace and an independently returned live-stage trace disagree at
352 U faces and 354 V faces by `8.881784197001252e-16`, and removing the trace
callback does not remove that disagreement. The apparent two-face boundary is
below this demonstrated compilation floor. It is not promoted as a citable
live boundary and cannot justify a numerical edit.

On NEMO's recorded inputs, the one shared production metric statement and the
one shared left-associated accumulator statement are bit-exact for all 50 U
substeps. The compiled accumulator is
`un_adv = un_adv + za2 * zhU * r1_e2u` at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:571-580`.
Thus neither shared downstream statement owns the robust gap.

The first robust, citable live boundary remains round 73's final kt=2
`un_adv`: 580 of 580 wet U faces, maximum absolute difference
`0.00012029895814569258`; that record's compiled finalization, boundary
exchange, and handoff are at
`GYRE_OMIP_L2_P3_SM_R72ZFOP/BLD/ppsrc/nemo/dynspg_ts.f90:797-817`.
The next compiled producer to test is the midpoint
extrapolation
`ua_e = za1*un_e + za2*ub_e + za3*ubb_e`, evaluated at
`GYRE_OMIP_L2_P3_SM_R75ADV3/BLD/ppsrc/nemo/dynspg_ts.f90:484-493`.
Its raw operands were not in any admitted record, so the walk stops there.

## Measurement evidence and controls

The admitted round-75 record is exactly 3,789,976 bytes, has SHA-256
`47ba91919df3a9a1c55be7fd886d620c721b2f175a737f32a05d1be2498ddf4c`,
and was produced at clean commit
`6ca9a954e84ae9cdef4885ae5c5b4f1e8fb68cf6`. Its admission reports 45 of 68
fields exact, 23 changed, and all 167 required consumed fields admitted.

The first measurement artifact is
`round76/round76_advmean_walk.json`, SHA-256
`a98f6d0a5244468b0fa8b5516527943ac6707bb58a3b7473db93ffa65cdb1cd0`.
It exited nonzero with status REFUTED. The callback-free retry is
`round76/round76_advmean_walk_retry.json`, SHA-256
`d743fc44488a26c87311ddfd8e5f930ce020865d53eb344b9ae37a78a1f721e4`;
it reproduced every decisive value and also exited nonzero as REFUTED. Every
record, live trace, and grid metric named in the result is float64. V remained
withheld after the unresolved U owner.

The non-vacuity controls both fired and exited 1. The null-live-`zhU` plant
moved the boundary and printed `ROUND76 NULL_LIVE_ZHU_PLANT FIRED`; its JSON
SHA-256 is
`2d7f2056ccd3ae5b4770fd37962af3200d5e8b031f7ccd3146b33f25edd319d6`.
The one-ULP recorded-exit plant printed `REFUSE: compiled advective-mean replay
is not bit-exact`; its log SHA-256 is
`7b995503a43ef2ad4c4239018756ea290f6192d9eea09f239607ec96c3faa99d`.

## Acquisition prepared, not run

The first acquisition preregistration proposed four full 36 by 26 fields and
1,499,040 bytes. That prediction is **RETRACTED before implementation** because
it could serialize unowned halo values. The committed correction freezes the
owned `ntsi:ntei,ntsj:ntej` extent: 32 by 22 values, one 16-byte magic, ten
int32 header values, and, for each of 50 substeps, `jn`, three float64
coefficients, and four float64 slices (`un_e`, `ub_e`, `ubb_e`, `ua_e`). The
exact corrected size is 1,127,856 bytes.

The new operator-only target is `GYRE_OMIP_L2_P3_SM_R76UAMID4`, derived from
the unchanged `GYRE_OMIP_L2_P3_SM_R75ADV3` source card. Its additive patch adds
only a WRITE-only kt=2 stream immediately after the compiled midpoint
statement. The source-run namelist must compare byte-identically, and the new
record is registered at the `now` time level. No canonical NEMO file is
modified.

The first preflight was **REFUTED** at exit 126 because `run.sh` lacked its
executable bit. Commit `7d1613a0350595beb967736b4d0e9d837b292de3` corrected that
packaging error. The clean retry performed exact preprocessing with the R75
keys and includes, ran `gfortran -fsyntax-only`, independently checked the
writer/reader layouts and exact byte count, printed
`ROUND76_UAMID_PREFLIGHT_READY`, and exited 0. Its log SHA-256 is
`f76b05102a6372673c69ad97415c86e7e5c9941b65f15c8d21a14bb1503c35ed`.
The final whole-round whitespace audit then found three trailing spaces in
context lines of the patch artifact. Commit
`0e59eed390fc855ca2dca5d41b14b2455af66edc` removed them and moved only the
local unit declaration to an equivalent clean context. The patch dry-applied,
and the clean-commit exact-preprocessing/Fortran preflight was repeated at that
commit; it produced the same readiness line and the same SHA-256 above.

The layout plant removed `ubb_e`, printed `REFUSE: layout plant removed the
ubb_e field`, and exited 69; its log SHA-256 is
`cab56e38981969df61830ab270e8adfe43ea1528529bf4861124bf35522559c6`.
The resolved-row plant printed `REFUSE: source run lacks resolved row: in
iterations nn_e *= *50` and exited 65; its log SHA-256 is
`32f9dedbb636839a7c6a95fbe63f52371ede4e43a2323570e798e2dcf8e8b544`.
The record-dependent header, truncation, replay-ULP, stamp, and consumed-field
plants are fail-closed in the gate/run card but remain UNREACHED until the
operator creates the record. The card itself requires every one to exit
nonzero before printing readiness.

No `makenemo` or `mpirun` was executed in this round. The acquisition path for
the operator is
`/tmp/autopilot-work-zAo44kg91P/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round76_uamid/run.sh`.

## Rule 12 disposition

No numerical statement was changed, so there is no candidate arm and no row
may move. The recorded round-73/round-75 trajectories remain the before arm;
scratch toggles are not substituted.

| Testcase / horizon | Executed statement | Mechanical disposition |
|---|---:|---|
| GYRE kt=1--10 | none | **UNREACHED / unchanged.** Ladder not rerun because no production candidate exists; every AT-BAR row and the first-over-bar boundary remain unchanged. |
| GYRE days 1--30 | none | **UNREACHED / unchanged.** Year member/owner gates not rerun because no production candidate exists; every registered day remains on the recorded before arm. |
| LOCK_EXCHANGE | none | **UNREACHED / unchanged.** There is no landed shared statement whose execution must be established. |
| OVERFLOW | none | **UNREACHED / unchanged.** There is no landed shared statement whose execution must be established. |
| DINO | none | **UNREACHED / explicit risk.** A future midpoint/shared-statement edit could affect DINO, and its 96--98% per-row cancellation makes aggregate neutrality insufficient. |
| ORCA2 | none | **UNMEASURED WITH SPEC.** Resolve its compiled external-mode card; record every entry, coefficient, midpoint operand/result, metric transport, reciprocal metric, accumulator exit, normalization, and boundary handoff for kt=1--10; replay in compiled order; register every moved row; preserve every AT-BAR row; and forbid an earlier first-over-bar boundary. |

No configuration/default, carried state, stabilizer, production ocean
statement, NEMO source/build/run, year harness, reconciliation gate,
freshwater pair, #1484 guard, or held manifest changed. The round-70 patch
remains held.

## Review and focused checks

The required separate Codex pass was attempted three times with `codex exec
--sandbox read-only` against the complete committed round diff and an
adversarial prompt covering the trace floor, source-order ownership, record
layout, plants, stamps, and every Rule 12 row. All attempts exited 1 before
review because the client could not initialize in this filesystem sandbox.
Its terminal result, quoted verbatim, is: **“Error: failed to initialize
in-process app-server client: Read-only file system (os error 30)”**. There is
no SHIP or DO NOT SHIP verdict; review is **UNMET/BLOCKED**, and absence of a
verdict is not approval. All three logs have SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No production numerical diff is landing under the blocked review.

The first focused run was **REFUTED**: the two new multi-line citation entries
anchored the lines preceding their terminal endpoints, and the clean-stamp
tests also refused the resulting dirty tree. It reported 48 passed and 4
failed. The endpoint occurrences were corrected and committed. The clean
retry reported 52 passed in 1.68 seconds; its JUnit SHA-256 is
`21c27ffa31357839fa1760070d5db4006d4df8a98c29e7b2931cca206f57bb53`.
It covers the inherited round-73 record gate, both round-76 instruments, the
whole citation-gate regression, and the time-level registry.

The receipt citation gate found four mapped compiled-source citations and no
failures at clean receipt commit `bc4e7c53504f`; its JSON SHA-256 is
`8e5ff21c1a04ac842dfdebed5a21dc6eec9ab6b2f945ad7f0ca5b1399802ebe3`.
Shifting the midpoint citation by two lines produced
`SYMBOL-NOT-AT-LINE` and exited 1; its log SHA-256 is
`c6db2004e99ceb76840348661292392cea5a0103c680cd83fd8d824f48f7c781`.
Shell parsing, Python compilation, focused Ruff, and `git diff --check` pass.

## Asked, unasked, and OPEN

**ASKED:** walk the first non-bit U statement in magnitude order, refuse an
under-instrumented conclusion, cite the compiled branch, preserve failed
predictions, prepare the missing acquisition, and apply Rule 12 mechanically.

**UNASKED:** no V walk, midpoint numerical patch, tolerance relaxation,
configuration decision, stabilizer, carried-state change, NEMO execution,
year-harness edit, freshwater edit, #1484 edit, or held-patch promotion was
performed.

**OPEN for round 77:** the operator must run the acquisition above. The next
round must first admit its clean commit stamp, exact 1,127,856-byte layout,
source/target restart and mesh identity, namelist identity, all 50 midpoint
replays, and every planted failure. Then read target
`GYRE_OMIP_L2_P3_SM_R76UAMID4`'s compiled `dynspg_ts.f90`, cite the live branch,
and run the one shared midpoint extrapolation on NEMO's raw inputs in exact
compiled association. Stop at its first non-bit operand or result. V remains
withheld until U is owned. No production edit is eligible without a frozen
Rule 12 moved-row table, the GYRE kt=1--10 ladder, GYRE days 1--30 owners,
LOCK_EXCHANGE and OVERFLOW execution proofs, explicit DINO risk, and the ORCA2
spec above.
