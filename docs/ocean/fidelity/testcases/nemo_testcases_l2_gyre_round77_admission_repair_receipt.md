# NEMO-testcases L2 GYRE round 77 receipt: acquisition admission repair

Date: 2026-09-12. Branch `fidelity/nemo-testcases-l2-gyre-codex2`.

## Verdict and first non-bit statement

**STOPPED FOR RECORD; NO PRODUCTION PHYSICS LANDED.** The operator's round-76
run successfully produced the requested midpoint record. The dedicated gate
accepted its exact 1,127,856-byte layout and all 50 midpoint replays before the
overall script stopped later in inherited-record admission. Therefore the old
byte-layout concern is **RETRACTED/RESOLVED**; the valid four-field, 32 by 22,
float64 writer and reader were not redesigned in this round.

The stop was an admission-instrument coverage error, not a model result. The
inherited kt=2 transport-mean record differed in raw halo storage, but the
shared admission checker had no parser for its `NEMO_L2_BTADV_2` magic and
exited 2 at the first changed record. The compiled target opens and heads that
inherited stream at
`GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:442-452`, then
writes its accumulator entry, metric transport, velocity, face depth, and exit
fields in branch-resolved order at `:588-606`.

Round 73's first robust scientific boundary remains the final kt=2 stage-1
`un_adv`: 580 of 580 wet U faces differ, with maximum absolute difference
`0.00012029895814569258`. The next scientific statement remains the compiled
midpoint association, which forms `ua_e` from `un_e`, `ub_e`, and `ubb_e` at
`GYRE_OMIP_L2_P3_SM_R76UAMID4/BLD/ppsrc/nemo/dynspg_ts.f90:501-509`.
This acquisition-only round did not walk or change that numerical statement.

## Operator record and corrected admission

The round-76 U-midpoint payload is exactly 1,127,856 bytes with SHA-256
`efff2a6ab74770890221790ac7b3c7d01d07bbf9ee3a520fdad1d99041be8a0b` and
clean producer commit `b7e6efc26a4e355c29541051d1ff9aae437b742f`. Its validation
JSON has SHA-256
`a899bc01b64dd229febc291945613af19088cffced2d4e3d3dab783bd9ab655e` and
status AT-BAR. The compiled target separately opens its reduced-domain stream
with explicit full and owned bounds at `:454-462`, then writes all four operands
immediately after the numerical association at `:501-509`.

The repaired shared admission checker composes the campaign's strict
transport-mean reader with an optional full-array view. It compares both
headers, the divisor, both 50-value weight sequences, both reciprocal metrics,
all ten named arrays at each of 50 substeps, and all four final pre/post-LBC
arrays. Scalar differences and owned array differences are violations; only
the explicit two-cell halo can be admitted, with every difference counted and
representative values printed.

At clean repair commit `49f9fe447c8050feb0652713c9465de3c8a469d6`, replaying the
completed round-76 source/candidate pair passes: 69 inherited records are
inventoried, 45 are raw byte-identical, 24 are classified, 281 representative
differences are reported under existing registered grounds, and both the final
restart and mesh mask are byte-identical. The previously unregistered kt=2
transport-mean row has 1,900 differing bytes representing 500 differing
float64 elements; all 500 are in the halo and zero are owned. Its JSON and log
SHA-256 values are respectively
`1aa3d9993f44db465d8f43d3b78b0f5fc03289b8fa906ffa81c12aec90e3d6b6` and
`222196ba2a24bf626388e8cef982bd44b828fe856dbf07b2f86a94ed0b060a0e`.

The preregistered owned-cell plant lands in the first transport-mean record,
makes that record fail consumed-field equality, prints
`CONSUMED_FIELD_ADMISSION FAIL`, and exits 1. Its JSON and log SHA-256 values
are respectively
`c4c8522dae7de1949dfd12563c96e0c811628ca8930e196de8a6a7cfad9e9d30` and
`cdf088cacc09326d848ef61587fed52b6661fffa96b0f1024ac55245dc97439e`.
The frozen prediction is **CONFIRMED**.

Two initial synthetic test attempts are retained as instrument corrections,
not scientific retractions. The first encoded the 15-character magic without
Fortran's trailing blank, and the second used an unregistered candidate file
name that correctly triggered the time-level registry. Padding the exact
16-byte magic and using the registered record name corrected the fixtures; the
same owned/halo assertions then passed.

## Replacement acquisition card

The existing operator card now selects new, absent target
`GYRE_OMIP_L2_P3_SM_R77UAMID5` and new, absent output
`round77/oracle_uamid_kt2`. It still starts from the unchanged R75 source card,
copies EXP00 and MY_SRC file by file, requires byte-identical `namelist_cfg`,
applies only the previously validated WRITE-only patch, and stamps the clean
commit that actually runs it. It never modifies canonical NEMO source.

The writer/reader preflight independently retains one 16-byte magic, ten int32
header values, 50 substeps, three float64 coefficients per substep, and four
32 by 22 float64 slices per substep, for exactly 1,127,856 bytes. Exact
preprocessing with the R75 compile keys and includes followed by
`gfortran -fsyntax-only` exited zero with empty compiler stdout and stderr. The
clean preflight printed `ROUND77_UAMID_PREFLIGHT_READY`; its log SHA-256 is
`f8e36e95c48a993f8dceaf77b97bf83545b806ca8e731e9398ef40debda819bd`.

The static layout plant removed `ubb_e`, printed the named refusal, and exited
69; log SHA-256
`cd783f65e9ced37b92818edd77f675201bd481b26b642fb15da9b4c33297ed93`.
The resolved-row plant changed the required 50-substep source evidence, printed
the missing-row refusal, and exited 65; log SHA-256
`7237742ae29f0bfddd2168a57824d5a6a727f25630fdbf75370f18df6602ef92`.
The record-dependent header, truncation, replay-ULP, stamp, consumed-field, and
inventory plants remain fail-closed in the operator path and are **UNREACHED**
for the new target until the operator creates it. The completed R76 replay
already proves the corrected consumed-field plant exits nonzero.

No `makenemo` or `mpirun` was invoked in this round. The operator acquisition
path is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round76_uamid/run.sh`.

## Rule 12 disposition

No numerical statement, configuration, default, or carried state changed, so
there is no candidate physics arm and no registered scientific row may move.
The recorded round-73/round-75 trajectories remain the before arm.

| Testcase / horizon | Executed changed production statement | Mechanical disposition |
|---|---:|---|
| GYRE kt=1--10 | none | **UNREACHED / unchanged.** The ladder was not rerun because no numerical candidate exists; every AT-BAR row and first-over-bar boundary remain unchanged. |
| GYRE days 1--30 | none | **UNREACHED / unchanged.** The year owner gate was not rerun because no numerical candidate exists; the recorded before arm remains authoritative. |
| LOCK_EXCHANGE | none | **UNREACHED / unchanged.** No shared production statement changed. |
| OVERFLOW | none | **UNREACHED / unchanged.** No shared production statement changed. |
| DINO | none | **UNREACHED / explicit risk.** A future midpoint edit is shared, and DINO's 96--98% per-row cancellation makes aggregate neutrality insufficient. |
| ORCA2 | none | **UNMEASURED WITH SPEC.** Resolve its compiled external-mode card; record every entry, coefficient, midpoint operand/result, metric transport, reciprocal metric, accumulator exit, normalization, and boundary handoff for kt=1--10; replay in compiled order; register every moved row; preserve every AT-BAR row; and forbid an earlier first-over-bar boundary. |

No stabilizer, production ocean statement, scientific configuration, canonical
NEMO source/build/run, year harness, reconciliation gate, freshwater pair,
#1484 guard, or held manifest changed. The round-70 patch remains held.

## Review and focused checks

The required separate Codex pass used `codex exec --sandbox read-only` against
the complete committed implementation diff and explicitly tried to refute the
record layout, owned/halo coverage, plant, acquisition target, commit stamp,
and Rule-12 table. It exited before review because its app-server client could
not initialize in the read-only sandbox. Its terminal result, quoted verbatim,
is: **“Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)”**. There is no SHIP or DO NOT SHIP verdict; review is
**UNMET/BLOCKED**, not treated as approval. No production numerical diff is
landing. The review log SHA-256 is
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.

The focused admission, transport-mean record, and U-midpoint record suites
report 27 passed in 1.51 seconds. Their JUnit SHA-256 is
`a118d2d9d0bd9844bed4d4e8b5a4823b187733d4eb5890c2ef0bd4401e77a865`.
Shell parsing, Python compilation, focused Ruff with the inherited long-line
and legacy test-name exceptions, and `git diff --check` pass.

The first citation run was **REFUTED**: both writer ranges ended on the
following `ENDIF`, while their registered terminal symbols were one line
earlier. The gate reported two `SYMBOL-NOT-AT-LINE` failures. Commit
`90b52cfbe2c4` shortened each range to its actual terminal statement. At that
clean commit the receipt gate found four citations, all mapped, and passed; its
JSON/log SHA-256 is
`0f3a099518d415d27d5c6ef504d8e264c0cbf8d9cbfc4672dfb0d376ea8d7703`.
Shifting the midpoint-association citation by two lines produced
`SYMBOL-NOT-AT-LINE` and exited 1; plant log SHA-256
`c2eea00a72b6c3eb252ead994115505ff2e46e671e13600fb1f8aa44b819c41d`.

## Asked, unasked, and OPEN

**ASKED:** consume the operator failure, distinguish a writer/reader size defect
from a later gate defect, repair the fail-closed acquisition path with a new
target, prove compilation, and stop for acquisition. Completed: the record was
already exact; the missing inherited-record comparator was the actual stop.

**UNASKED:** no scientific midpoint walk, V walk, numerical patch, tolerance
relaxation, configuration decision, stabilizer, carried-state change, NEMO
execution, year-harness edit, freshwater edit, #1484 edit, or held-patch
promotion was performed.

**OPEN for round 78:** the operator must run the replacement acquisition. The
next round must first require its clean commit stamp, exact 1,127,856-byte
layout, source/target restart and mesh identity, namelist identity, all 50
midpoint replays, full inherited-record admission, and every planted failure.
Then read and cite the new target's compiled `dynspg_ts.f90`, feed NEMO's raw
`un_e`, `ub_e`, `ubb_e`, and coefficients through the one shared legoESM
midpoint statement in exact compiled association, and stop at its first non-bit
operand or result. V remains withheld until U is owned. No production edit is
eligible without a frozen Rule-12 moved-row table, GYRE kt=1--10 ladder, GYRE
days 1--30 owners, LOCK_EXCHANGE and OVERFLOW execution proofs, explicit DINO
risk, and the ORCA2 spec above.
