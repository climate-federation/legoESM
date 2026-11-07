# NEMO testcase L2 GYRE round 148 — developed LDF acquisition receipt

Date: 2026-09-22  
Incoming lane tip: `973ec1c51ca1880ef6c158cbbe3c6a7282e9315a`  
Preregistration commit: `d86158459`  
Acquisition implementation commit: `dbdd18e4a`  
Preflight repair commit: `4bfa36dca`  
Citation-map commit: `69532b2b9`  
Status: **STOPPED_FOR_RECORD — the existing records do not contain the direct
LDF operands or intermediates; a syntax-proven passive acquisition is ready**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round148/`.

## Result

No scientific LDF operand row was measured and no first non-bit internal
statement is named.  Round 147's cumulative HPG/LDF snapshots can identify the
LDF family, but subtraction cannot recover the isolated LDF addend after the
shared accumulator has rounded.  The admitted developed RHS record also lacks
the six live thickness operands and the compiled curl/divergence
intermediates.  Treating either reconstruction as an oracle would violate the
frozen discrimination.

The compiled routine first forms the F-point `zwf` and T-point `zwt` values,
then updates U/V in place at
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-140`.
The external developed-state caller binds both formal thickness levels to
`Kbb` at
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/stp2d.f90:145-192`.
The new writer is therefore gated on `kt=1081` and `Kbb==Kmm`; its header and
pre/post family-boundary checks refuse a record from any other call context.

The frozen prediction remains **UNMEASURED**: the first source-order non-bit
operand will be NEMO's live F-point thickness.  It is explicitly refuted if
that field is BIT or if an earlier recorded input differs.

## Acquisition contract

The new target is `GYRE_OMIP_L2_P3_SM_R148LDF`, cloned file by file from the
admitted Round-146 producing card.  The additive source card changes no NEMO
statement.  At the single external LDF call it records a closed 30-field
stream:

| group | fields | count |
|---|---|---:|
| exact inputs | U/V, T/U/V/F masks, `ahmt`/`ahmf` | 8 three-dimensional |
| live thickness | `e3t(Kbb)`, `e3u/e3v(Kbb)`, `e3f`, `e3u/e3v(Kmm)` | 6 three-dimensional |
| accumulator | U/V before and after LDF | 4 three-dimensional |
| compiled intermediates | `zcur`, `zdiv` | 2 three-dimensional |
| horizontal metrics | four direct metrics and six reciprocals | 10 two-dimensional |

The exact record size is `4,717,612` bytes.  The extended Round-50 reader
requires the header, all 30 sizes, all finite values, EOF, SHA/commit stamp,
and BIT closure of pre-LDF U/V to Round 146's HPG boundary and post-LDF U/V to
its LDF boundary on every native wet face.  It registers 26,250 owned values
for each direct intermediate (35 x 25 x 30); no uninitialized halo is used as
scientific evidence.

Admission additionally requires byte identity for both restarts and the
inherited process-budget, external-step, QCO, slow-forcing, completed-RHS, and
RHS-family records.  Any movement refuses the record.

## Preflight and controls

The source card applies with zero removed lines.  Preprocessing the exact card
with the producing build's keys followed by `gfortran -fsyntax-only` prints:

`SYNTAX_PROOF_PASS dynldf_lev.f90`

The complete preflight prints
`ROUND148_DEVELOPED_LDF_PREFLIGHT_READY`.  The writer/reader byte counts agree.
The layout plant removes the unique completion marker, prints
`STATUS PLANT-FIRED: layout`, prints a named `REFUSE`, and exits 69.

The first preflight attempt is retained as `preflight_failed_import.log`.  It
failed before source preprocessing because the extended walker's sibling
reader directory was absent from `PYTHONPATH`.  Commit `4bfa36dca` adds that
existing directory; the successful preflight above follows.  This is an
instrument repair, not a scientific result or retraction.

The parser's synthetic record test covers ordinary admission plus header,
truncation, missing-field, direct-`zcur` ULP, and post-accumulator ULP plants.
The acquisition script reruns all five against the real stamped record after
NEMO finishes; each must print `STATUS PLANT-FIRED` and exit nonzero.

## Review

The required separate read-only Codex review was attempted against the
committed Round-148 diff.  Independent review was unavailable in-sandbox; its
verbatim terminal finding was:
`Error: failed to initialize in-process app-server client: Read-only file
system (os error 30)`.  This is not represented as a SHIP verdict.

## Frozen trajectory and scope

No production code, physics, configuration, default, carried state, scheme,
stabilizer, canonical NEMO source, or immutable before arm changed.  No NEMO
integration was attempted in the sandbox, per the standing PMIx instruction.
The headline remains:

| row | unchanged value |
|---|---:|
| kt2 U RMS | `2.7377110452773967e-12` |
| kt2 V RMS | `3.2849219221489645e-12` |
| kt3 T RMS | `8.659373840202989e-7 K` |
| kt3 S RMS | `7.027291104577671e-8` |
| day-30 T3D RMS | `6.890431487825909e-5 K` |
| day-240 T3D RMS | `1.644674023317539e-2 K` |
| day-360 T3D RMS | `1.122357124784366e-2 K` |

DINO shares the operator and must be measured if the direct record produces a
candidate.  LOCK_EXCHANGE, OVERFLOW, and the tanks execute no changed
production statement.  ORCA2 is `UNMEASURED-WITH-SPEC`: its own developed
state needs this same operand/intermediate record before any cross-card exactness
claim.  No configuration choice was made.

## Evidence and tests

The final focused extended-walker and citation-gate suite reports
`20 passed in 8.11s`.  The preregistration citation gate passes with two mapped
citations and no failures; the receipt gate passes with three mapped citations
and no failures;
shifting the
`GYRE_OMIP_L2_P3_SM_R146RHSFAM/BLD/ppsrc/nemo/dynldf_lev.f90:121-130`
citation by two lines makes it report
`SYMBOL-NOT-AT-LINE` and exit 1.  Final focused and receipt-citation results are
recorded below after the receipt commit.

| artifact | SHA-256 |
|---|---|
| `preflight.log` | `f8fd9f78d1fa3d8c7b6fe4171a0adc68ba91d8c80b05a1d0056c861a5b8cab66` |
| failed `preflight_failed_import.log` | `93dcaddaedd0546b4cf909e15e66e226fe738937fec25c5502c62ab9004f53cc` |
| `layout_plant.log` | `59c746863b621b902b93c83d5fe391707fe973829792945b9706167048cdf1b5` |
| `prereg_citation_gate.json` | `a98dda6b23dfc93f13948cd2d28827aab6a3e276242b948dd25772090691a657` |
| `prereg_citation_plant.json` | `dc9fcac113f2a6b07b6c98791a59f434790b566dc4fc999ac3bb675e2b28279d` |
| `receipt_citation_gate.json` | `dc8fecd7eb3b0e83d75f470ed60ec890fbf41b13eda02bb6f4c49d709f685cd0` |
| `receipt_citation_plant.json` | `32ed5136295b7aa4a5526aa22a163bc4e6785af3f10d76652c7a301981071600` |
| `focused_tests_final.log` | `a28ac4c3b08e04414ef1a51c11044362dcfa40fa09bf55e68908fd63f8e7ce4f` |
| `codex_review.log` | `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5` |

## OPEN — round 149

Run the committed acquisition script.  It must print
`ROUND148_DEVELOPED_LDF_READY` after all inherited artifacts remain BIT and all
admission plants fire.  Then extend the same Round-50 walker through the
production-JIT developed step: reproduce Round 147's LDF-family row, score the
recorded inputs and `zcur`/`zdiv` in compiled order, distinguish the isolated
operator from the pre/post accumulator, and name the first non-bit statement.
Only a source-exact candidate proceeds to the complete Decision-43/45 ladder,
month, year, and DINO gates.

`ACQUISITION_NEEDED`:
`/tmp/autopilot-work-e480ZvF6/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round148_ldf/run.sh`  
`DECISION_NEEDED`: NONE.  
`ROUND_STATUS`: STOPPED_FOR_RECORD.
